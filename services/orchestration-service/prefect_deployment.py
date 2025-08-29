from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from flows import (
    story_to_video_pipeline,
    content_generation_flow,
    video_assembly_flow,
    distribution_flow,
    analytics_collection_flow,
)

# Example deployments
story_to_video_deploy = Deployment.build_from_flow(
    flow=story_to_video_pipeline,
    name="story-to-video-production",
    schedule=CronSchedule(cron="0 */2 * * *"),  # Every 2 hours
    work_pool_name="ai-pipeline-pool",
)

content_generation_deploy = Deployment.build_from_flow(
    flow=content_generation_flow,
    name="content-generation-hourly",
    schedule=CronSchedule(cron="0 * * * *"),  # Hourly
    work_pool_name="ai-pipeline-pool",
)

analytics_collection_deploy = Deployment.build_from_flow(
    flow=analytics_collection_flow,
    name="analytics-collection-daily",
    schedule=CronSchedule(cron="0 3 * * *"),  # Daily at 03:00 UTC
    work_pool_name="ai-pipeline-pool",
)

if __name__ == "__main__":
    story_to_video_deploy.apply()
    content_generation_deploy.apply()
    analytics_collection_deploy.apply()
    print("Prefect deployments applied.")
