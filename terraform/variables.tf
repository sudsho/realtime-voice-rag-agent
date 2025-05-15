variable "region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project" {
  description = "project name used as a prefix"
  type        = string
  default     = "voice-rag"
}

variable "env" {
  type    = string
  default = "dev"
}

variable "image_uri" {
  description = "ECR image URI for the agent container"
  type        = string
}

variable "desired_count" {
  type    = number
  default = 1
}

variable "openai_secret_arn" {
  description = "Secrets Manager ARN that holds OPENAI_API_KEY"
  type        = string
}
