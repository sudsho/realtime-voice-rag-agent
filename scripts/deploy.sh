#!/usr/bin/env bash
# Build, push, and roll the ECS service for the voice agent.
#
#   ./scripts/deploy.sh staging
#   ./scripts/deploy.sh prod
#
# Required env: AWS_REGION, AWS_ACCOUNT_ID, ECR_REPO, ECS_CLUSTER, ECS_SERVICE
set -euo pipefail

ENV_NAME="${1:-staging}"
TAG="$(git rev-parse --short HEAD)"
IMAGE="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${TAG}"

echo ">> building ${IMAGE}"
docker build -t "${IMAGE}" .

echo ">> ecr login"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo ">> pushing image"
docker push "${IMAGE}"

echo ">> applying terraform (env=${ENV_NAME})"
pushd terraform >/dev/null
terraform init -input=false -upgrade=false
terraform apply -auto-approve \
  -var "env=${ENV_NAME}" \
  -var "image_uri=${IMAGE}"
popd >/dev/null

echo ">> forcing new deployment"
aws ecs update-service \
  --cluster "${ECS_CLUSTER}" \
  --service "${ECS_SERVICE}" \
  --force-new-deployment \
  --region "${AWS_REGION}" >/dev/null

echo ">> waiting for service to stabilize"
aws ecs wait services-stable \
  --cluster "${ECS_CLUSTER}" \
  --services "${ECS_SERVICE}" \
  --region "${AWS_REGION}"

echo ">> done. deployed ${IMAGE} to ${ENV_NAME}."
