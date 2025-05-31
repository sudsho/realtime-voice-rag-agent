# deploy runbook

low-touch deploy for the voice RAG agent on AWS (ECS Fargate behind ALB,
fronted by CloudFront for the static frontend + WS upgrade).

## prerequisites

- aws cli logged in to the target account
- terraform >= 1.6
- docker
- secrets in AWS Secrets Manager:
  - `voice-rag-agent/openai` -> JSON `{"OPENAI_API_KEY": "sk-..."}`

set the env vars below before running the deploy script:

```
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export ECR_REPO=voice-rag-agent
export ECS_CLUSTER=voice-rag-agent
export ECS_SERVICE=voice-rag-agent
```

## one-time bootstrap

```
aws ecr create-repository --repository-name "$ECR_REPO" --region "$AWS_REGION"
cd terraform && terraform init && terraform apply -var env=staging -var image_uri=public.ecr.aws/docker/library/hello-world:latest
```

(the placeholder image lets terraform create the cluster/service/ALB before
we have a real image; the first `deploy.sh` run will roll the real one.)

## deploy

```
./scripts/deploy.sh staging
# verify
curl https://<alb-dns>/healthz
# smoke the WS endpoint with the included frontend
open https://<cloudfront-domain>/
```

prod is identical with `prod` instead of `staging`.

## rollback

ECS keeps the previous task definition. To roll back:

```
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE \
  --task-definition <previous-revision> --region $AWS_REGION
```

or re-run `deploy.sh` with an older git sha checked out:

```
git checkout <previous-sha> -- .
./scripts/deploy.sh prod
git checkout main -- .
```

## sanity checks after rollout

1. `/healthz` and `/readyz` return 200
2. CloudWatch logs show structured JSON with `LOG_FORMAT=json`
3. open the frontend, speak, confirm:
   - first audio chunk arrives within ~300ms after the LLM starts streaming
   - barge-in interrupts ongoing speech
4. CloudWatch metric `WSConnections` and `TurnLatencyMs` are flowing

## scaling notes

- service is sized for ~50 concurrent sessions per task (CPU-bound on STT)
- horizontal autoscaling on `CPUUtilization` target 60% via terraform
- bump `desired_count` for a hard floor before a known burst
