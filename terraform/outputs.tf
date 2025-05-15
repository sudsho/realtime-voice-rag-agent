output "alb_dns_name" {
  value = module.ecs.alb_dns_name
}

output "cloudfront_domain" {
  value = module.cdn.cloudfront_domain
}

output "ecr_image" {
  value = var.image_uri
}
