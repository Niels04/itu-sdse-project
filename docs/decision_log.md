# Decision log

One entry per decision. Newest at the bottom.

## Meeting-01: One shared repository
Status: Accepted · Date: 2026-11-05
Context: Four people hand in one repository with its full history.
Decision: Everyone works in Nielson's fork (Niels04). Changes go through a pull request with one approval, and the author merges with a merge commit.
Consequences: Tasks must be divided, and everyone reviews each other.
We decide to meet every Thursday from 10:30AM. Room is booked every Thursday 10AM to 2pm.

## Meeting-02: Host the model on Azure instead of DigitalOcean
Status: Accepted · Date: 2026-11-19
Context: The teachers moved the course hosting from DigitalOcean to Azure (upstream #94, merged here as #15). The DigitalOcean app spec no longer matched our deployment.
Decision: We deploy the serving image from GHCR to Azure Container Apps, using `infra/azure/app.yaml` and `az_setup.md`. The DigitalOcean spec (`infra/digitalocean/app.yaml`) is removed.
Consequences: Azure needs a GHCR token to pull the image (#18). The deploy step must run `az containerapp update` with the new image tag, and `PROD_ENDPOINT` in `serve.py` and `export.py` must point to the Azure URL once the app is running.
