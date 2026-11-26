// pipeline.go runs the project's pipeline in containers with Dagger.
// Run it from the project root: go run pipeline.go
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"dagger.io/dagger"
)

const baseImage = "ghcr.io/astral-sh/uv:python3.13-bookworm"

func main() {
	ctx := context.Background()
	if err := run(ctx); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// run connects to Dagger and chains the stages in order.
func run(ctx context.Context) error {
	client, err := dagger.Connect(ctx, dagger.WithLogOutput(os.Stderr))
	if err != nil {
		return fmt.Errorf("connect to Dagger: %w", err)
	}
	defer client.Close()

	base := baseContainer(client)
	tested := unitTestStage(base)
	withData := dataStage(tested)

	trained := trainStage(withData)
	exported := exportStage(trained)
	modelTested := predictStage(exported)

	// Nothing runs until a result is requested; Sync runs all stages above.
	if _, err := modelTested.Sync(ctx); err != nil {
		return fmt.Errorf("pipeline failed: %w", err)
	}

	// Copy the tested model and the results from the container to the host.
	if _, err := modelTested.Directory("/app/models/model").Export(ctx, "models/model"); err != nil {
		return fmt.Errorf("export model: %w", err)
	}
	if _, err := modelTested.Directory("/app/artifacts").Export(ctx, "artifacts"); err != nil {
		return fmt.Errorf("export artifacts: %w", err)
	}
	// Stop if the exported folder is not a complete MLflow model.
	if err := checkArtifact("models/model"); err != nil {
		return fmt.Errorf("pipeline failed: %w", err)
	}

	fmt.Println("Pipeline finished")

	// Here will be the pipeline part that produces the serving container:
	// (for now this just prints the image reference under which the image will be published to ghcr.io)

	// Only publish if on main branch:
	if os.Getenv("GITHUB_REF") == "refs/heads/main" {
		// get the sha of the current commit
		sha, err := currentCommitSHA()
		if err != nil {
			return fmt.Errorf("pipeline failed: %w", err)
		}
		// construct the image reference from the sha
		imgRef, err := imageRef("Niels04", sha)
		if err != nil {
			return fmt.Errorf("pipeline failed: %w", err)
		}

		fmt.Println(imgRef) // <- replace this with publishing the container
	}

	return nil
}

// -------- PIPELINE STEPS

// baseContainer copies the project into a Python container and installs
// exactly the packages locked in uv.lock.
func baseContainer(client *dagger.Client) *dagger.Container {
	source := client.Host().Directory(".", dagger.HostDirectoryOpts{
		Exclude: []string{".venv", "artifacts", "models/model", "mlflow-state", "data/images/raw", "data/test/*.jpg"},
	})
	return client.Container().
		From(baseImage).
		WithMountedCache("/root/.cache/uv", client.CacheVolume("uv-cache")).
		WithEnvVariable("UV_LINK_MODE", "copy").
		WithDirectory("/app", source).
		WithWorkdir("/app").
		WithExec([]string{"uv", "sync", "--frozen"})
}

// unitTestStage runs the tests in tests/test_units.py with pytest. (so only the unit tests)
func unitTestStage(c *dagger.Container) *dagger.Container {
	return c.WithExec([]string{"uv", "run", "pytest", "tests/test_units.py"})
}

// model prediction test stage runs all tests in tests/ with pytest.
func predictStage(c *dagger.Container) *dagger.Container {
	return c.WithExec([]string{"uv", "run", "pytest"})
}

// dataStage fetches the metadata files tracked by DVC.
func dataStage(c *dagger.Container) *dagger.Container {
	return c.WithExec([]string{"uv", "run", "dvc", "pull"})
}

// trainStage downloads the images and trains the model with the training script.
func trainStage(c *dagger.Container) *dagger.Container {
	return c.WithExec([]string{"uv", "run", "python", "vial_classifier/modeling/train.py"})
}

// exportStage registers the best run and exports the servable model to models/model.
func exportStage(c *dagger.Container) *dagger.Container {
	return c.WithExec([]string{"uv", "run", "python", "vial_classifier/modeling/export.py"})
}

// ------- UTILITY FUNCTIONS

// imageRef returns the name of our serving image on GHCR.
func imageRef(owner, sha string) (string, error) {
	if owner == "" || sha == "" {
		return "", errors.New("owner and sha must not be empty")
	}
	// GHCR only accepts lowercase names, so "Niels04" becomes "niels04".
	return "ghcr.io/" + strings.ToLower(owner) + "/model-serve:" + sha, nil

}

// currentCommitSHA() returns the SHA of the current git commit
func currentCommitSHA() (string, error) {
	if sha := strings.TrimSpace(os.Getenv("GITHUB_SHA")); sha != "" {
		return sha, nil
	}

	output, err := exec.Command("git", "rev-parse", "HEAD").Output()
	if err != nil {
		return "", fmt.Errorf("determine commit SHA: %w", err)
	}

	sha := strings.TrimSpace(string(output))
	if sha == "" {
		return "", fmt.Errorf("commit SHA is empty")
	}
	return sha, nil
}

// checkArtifact returns an error if dir does not contain an exported MLflow model.
func checkArtifact(dir string) error {
	if _, err := os.Stat(filepath.Join(dir, "MLmodel")); err != nil {
		return fmt.Errorf("no MLflow model in %s: %w", dir, err)
	}
	return nil
}
