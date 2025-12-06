#!/usr/bin/env python3
"""Query HuggingFace Jobs information: status, logs, details."""

import argparse
import json
from typing import Any

from huggingface_hub import HfApi


def get_job_info(api: HfApi, job_id: str) -> dict[str, Any]:
    """
    Get detailed information about a job.

    Args:
        api: HuggingFace API client
        job_id: Job ID to inspect

    Returns:
        Job information dictionary
    """
    job = api.inspect_job(job_id=job_id)
    return {
        "id": job.id,
        "status": job.status.stage if hasattr(job.status, "stage") else str(job.status),
        "status_message": job.status.message
        if hasattr(job.status, "message")
        else None,
        "flavor": job.flavor,
        "created_at": str(job.created_at) if job.created_at else None,
        "command": job.command,
        "arguments": job.arguments,
        "docker_image": job.docker_image,
        "owner": job.owner.name if hasattr(job.owner, "name") else str(job.owner),
        "url": job.url,
    }


def get_job_logs(api: HfApi, job_id: str, tail: int | None = None) -> str:
    """
    Fetch logs for a job.

    Args:
        api: HuggingFace API client
        job_id: Job ID
        tail: Number of lines from end (None for all)

    Returns:
        Log content as string
    """
    log_lines = list(api.fetch_job_logs(job_id=job_id))
    if not log_lines:
        return "(no logs available)"
    if tail:
        log_lines = log_lines[-tail:]
    return "\n".join(log_lines)


def list_jobs(api: HfApi, limit: int = 10, status: str | None = None) -> list[dict]:
    """
    List recent jobs.

    Args:
        api: HuggingFace API client
        limit: Maximum number of jobs to return
        status: Filter by status (running, completed, failed, etc.)

    Returns:
        List of job summaries
    """
    jobs = list(api.list_jobs())

    if status:
        jobs = [
            j
            for j in jobs
            if (j.status.stage if hasattr(j.status, "stage") else str(j.status)).lower()
            == status.lower()
        ]

    jobs = jobs[:limit]

    return [
        {
            "id": j.id,
            "status": j.status.stage if hasattr(j.status, "stage") else str(j.status),
            "flavor": j.flavor,
            "created_at": str(j.created_at) if j.created_at else None,
        }
        for j in jobs
    ]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Query HuggingFace Jobs information")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # info subcommand
    info_parser = subparsers.add_parser("info", help="Get job details")
    info_parser.add_argument("job_id", help="Job ID to inspect")
    info_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # logs subcommand
    logs_parser = subparsers.add_parser("logs", help="Fetch job logs")
    logs_parser.add_argument("job_id", help="Job ID")
    logs_parser.add_argument("--tail", type=int, help="Show last N lines")

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument("--limit", type=int, default=10, help="Max jobs to show")
    list_parser.add_argument(
        "--status", help="Filter by status (running, completed, failed)"
    )
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # cancel subcommand
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_id", help="Job ID to cancel")

    args = parser.parse_args()
    api = HfApi()

    try:
        if args.command == "info":
            info = get_job_info(api, args.job_id)
            if args.json:
                print(json.dumps(info, indent=2))
            else:
                print(f"Job: {info['id']}")
                print(f"  Status:     {info['status']}")
                if info["status_message"]:
                    print(f"  Message:    {info['status_message']}")
                print(f"  Flavor:     {info['flavor']}")
                print(f"  Created:    {info['created_at']}")
                print(f"  Owner:      {info['owner']}")
                print(f"  Command:    {info['command']}")
                print(f"  Arguments:  {info['arguments']}")
                print(f"  Image:      {info['docker_image']}")
                print(f"  URL:        {info['url']}")

        elif args.command == "logs":
            logs = get_job_logs(api, args.job_id, args.tail)
            print(logs)

        elif args.command == "list":
            jobs = list_jobs(api, args.limit, args.status)
            if args.json:
                print(json.dumps(jobs, indent=2))
            else:
                if not jobs:
                    print("No jobs found")
                else:
                    print(f"{'ID':<30} {'Status':<12} {'Flavor':<15} {'Created'}")
                    print("-" * 80)
                    for j in jobs:
                        created = j["created_at"][:19] if j["created_at"] else "N/A"
                        print(
                            f"{j['id']:<30} {j['status']:<12} {j['flavor']:<15} {created}"
                        )

        elif args.command == "cancel":
            api.cancel_job(job_id=args.job_id)
            print(f"Job {args.job_id} cancelled")

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
