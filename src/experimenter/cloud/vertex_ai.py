"""Google Cloud Vertex AI custom training job submission."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Pre-built containers (CPU for tabular, GPU for NLP)
CONTAINERS = {
    "tabular_cpu": "us-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-5:latest",
    "nlp_gpu": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-3:latest",
}

TRAINING_SCRIPT_DIR = Path(__file__).parent.parent.parent.parent / "training_scripts"


def submit_vertex_job(
    plan,  # ExperimentPlan
    data_gcs_uri: str,
    output_gcs_uri: str,
    wandb_api_key: str = "",
    wandb_project: str = "experimenter",
) -> str:
    """
    Submit a Vertex AI CustomJob and return the job resource name.

    Raises RuntimeError if GCP is not configured.
    """
    from experimenter.config import settings

    if not settings.has_gcp:
        raise RuntimeError(
            "GCP not configured. Set GCP_PROJECT_ID and GCS_BUCKET in .env"
        )

    try:
        from google.cloud import aiplatform

        aiplatform.init(project=settings.gcp_project_id, location=settings.gcp_region)

        is_nlp = plan.task_type in ("nlp_classification", "nlp_regression")
        container_uri = CONTAINERS["nlp_gpu"] if is_nlp else CONTAINERS["tabular_cpu"]
        script_name = "train_nlp.py" if is_nlp else "train_tabular.py"

        # Upload training script to GCS
        from experimenter.cloud.gcs import upload_file

        script_path = TRAINING_SCRIPT_DIR / script_name
        script_gcs = f"{output_gcs_uri}/scripts/{script_name}"
        upload_file(str(script_path), script_gcs)

        env_vars = {
            "WANDB_API_KEY": wandb_api_key,
            "WANDB_PROJECT": wandb_project,
        }

        args = [
            "--data-path", data_gcs_uri,
            "--target-column", plan.target_variable,
            "--task-type", plan.task_type,
            "--model-type", plan.model_type if plan.model_type != "auto" else "xgboost",
            "--output-dir", output_gcs_uri,
        ]

        job = aiplatform.CustomJob(
            display_name=f"experimenter-{plan.id[:8]}",
            worker_pool_specs=[
                {
                    "machine_spec": {
                        "machine_type": plan.machine_type,
                        **(
                            {"accelerator_type": "NVIDIA_TESLA_T4", "accelerator_count": 1}
                            if plan.needs_gpu
                            else {}
                        ),
                    },
                    "replica_count": 1,
                    "python_package_spec": {
                        "executor_image_uri": container_uri,
                        "package_uris": [],
                        "python_module": script_name.replace(".py", "").replace("/", "."),
                        "args": args,
                        "env": [{"name": k, "value": v} for k, v in env_vars.items()],
                    },
                }
            ],
        )

        job.submit(service_account=None, enable_web_access=False)
        logger.info("Submitted Vertex AI job: %s", job.resource_name)
        return job.resource_name

    except Exception as exc:
        logger.error("Vertex AI job submission failed: %s", exc)
        raise


def get_job_status(job_resource_name: str) -> str:
    """Return the Vertex AI job state as a string."""
    try:
        from google.cloud import aiplatform

        job = aiplatform.CustomJob.get(job_resource_name)
        return str(job.state.name).lower()
    except Exception as exc:
        logger.error("Failed to get job status: %s", exc)
        return "unknown"
