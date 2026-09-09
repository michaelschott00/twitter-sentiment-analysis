locals {
  tags = {
    project = "twitter-sentiment"
    env     = var.env
    owner   = var.owner
  }

  resource_group_name = "rg-twitter-ml"
  workspace_name      = "mlw-twitter-sentiment"
  law_name            = "log-twitter-ml"
  app_insights_name   = "appi-twitter-ml"
}
