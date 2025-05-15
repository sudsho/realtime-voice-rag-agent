terraform {
  required_version = ">= 1.6"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.80"
    }
  }
}

provider "aws" {
  region = var.region
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name = var.project
  tags = {
    Project = var.project
    Env     = var.env
    Owner   = "voice-rag-agent"
  }
}

module "network" {
  source        = "./modules/network"
  name          = local.name
  vpc_cidr      = "10.42.0.0/16"
  public_cidrs  = ["10.42.0.0/24", "10.42.1.0/24"]
  private_cidrs = ["10.42.10.0/24", "10.42.11.0/24"]
  azs           = slice(data.aws_availability_zones.available.names, 0, 2)
  tags          = local.tags
}

module "ecs" {
  source             = "./modules/ecs"
  name               = local.name
  vpc_id             = module.network.vpc_id
  private_subnet_ids = module.network.private_subnet_ids
  public_subnet_ids  = module.network.public_subnet_ids
  image              = var.image_uri
  container_port     = 8000
  cpu                = 1024
  memory             = 2048
  desired_count      = var.desired_count
  openai_secret_arn  = var.openai_secret_arn
  tags               = local.tags
}

module "cdn" {
  source       = "./modules/cdn"
  name         = local.name
  alb_dns_name = module.ecs.alb_dns_name
  tags         = local.tags
}
