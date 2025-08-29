"""
Batch Processing System for Image Generation

Handles asynchronous batch image generation with job queuing, status tracking,
and retry mechanisms using Redis as the backend.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchJob:
    """Batch job data structure"""
    job_id: str
    prompts: List[str]
    provider: str
    parameters: Dict[str, Any]
    story_id: Optional[str] = None
    user_id: Optional[str] = None
    priority: int = 0
    status: JobStatus = JobStatus.QUEUED
    progress: int = 0
    total: int = 0
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

    def __post_init__(self):
        if self.created_at == 0:
            self.created_at = time.time()
        if self.total == 0:
            self.total = len(self.prompts)
        if self.results is None:
            self.results = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BatchJob':
        """Create from dictionary from Redis"""
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = JobStatus(data['status'])
        return cls(**data)


class BatchProcessor:
    """Batch processing system for image generation"""

    def __init__(self, redis_client: redis.Redis, providers: Dict[str, Any]):
        self.redis = redis_client
        self.providers = providers
        self.processing = False
        self.worker_task = None
        self.max_concurrent_jobs = 3
        self.job_timeout = 3600  # 1 hour
        
        # Redis keys
        self.queue_key = "image_batch_queue"
        self.processing_key = "image_batch_processing"
        self.jobs_key = "image_batch_jobs"
        self.stats_key = "image_batch_stats"

    async def start_processing(self):
        """Start the batch processing worker"""
        if self.processing:
            return
        
        self.processing = True
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Batch processor started")

    def stop_processing(self):
        """Stop the batch processing worker"""
        self.processing = False
        if self.worker_task:
            self.worker_task.cancel()
        logger.info("Batch processor stopped")

    async def submit_batch_job(
        self,
        prompts: List[str],
        provider: str,
        parameters: Optional[Dict[str, Any]] = None,
        story_id: Optional[str] = None,
        user_id: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """Submit a new batch job"""
        job_id = str(uuid.uuid4())
        
        job = BatchJob(
            job_id=job_id,
            prompts=prompts,
            provider=provider,
            parameters=parameters or {},
            story_id=story_id,
            user_id=user_id,
            priority=priority
        )
        
        # Store job data
        await self.redis.hset(
            self.jobs_key,
            job_id,
            json.dumps(job.to_dict())
        )
        
        # Add to priority queue (higher priority = lower score)
        await self.redis.zadd(
            self.queue_key,
            {job_id: -priority}
        )
        
        # Update stats
        await self._update_stats("submitted", 1)
        
        logger.info(f"Batch job {job_id} submitted with {len(prompts)} prompts")
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[BatchJob]:
        """Get job status and results"""
        job_data = await self.redis.hget(self.jobs_key, job_id)
        if not job_data:
            return None
        
        return BatchJob.from_dict(json.loads(job_data))

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job if it's not already processing"""
        job = await self.get_job_status(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False
        
        # Remove from queue if queued
        if job.status == JobStatus.QUEUED:
            await self.redis.zrem(self.queue_key, job_id)
        
        # Update job status
        job.status = JobStatus.CANCELLED
        job.completed_at = time.time()
        
        await self.redis.hset(
            self.jobs_key,
            job_id,
            json.dumps(job.to_dict())
        )
        
        logger.info(f"Job {job_id} cancelled")
        return True

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        queue_length = await self.redis.zcard(self.queue_key)
        processing_count = await self.redis.scard(self.processing_key)
        
        # Get status counts
        all_jobs = await self.redis.hgetall(self.jobs_key)
        status_counts = {}
        
        for job_data in all_jobs.values():
            job = BatchJob.from_dict(json.loads(job_data))
            status = job.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "queue_length": queue_length,
            "processing": processing_count,
            "status_counts": status_counts,
            "total_jobs": len(all_jobs)
        }

    async def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed jobs"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        all_jobs = await self.redis.hgetall(self.jobs_key)
        
        cleaned = 0
        for job_id, job_data in all_jobs.items():
            job = BatchJob.from_dict(json.loads(job_data))
            
            # Only clean completed, failed, or cancelled jobs
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.completed_at and job.completed_at < cutoff_time:
                    await self.redis.hdel(self.jobs_key, job_id)
                    cleaned += 1
        
        logger.info(f"Cleaned up {cleaned} old jobs")
        return cleaned

    async def _worker_loop(self):
        """Main worker loop for processing batch jobs"""
        logger.info("Batch processor worker loop started")
        
        while self.processing:
            try:
                # Get next job from queue
                job_id = await self._get_next_job()
                if not job_id:
                    await asyncio.sleep(1)
                    continue
                
                # Process the job
                await self._process_job(job_id)
                
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _get_next_job(self) -> Optional[str]:
        """Get the next job from the priority queue"""
        # Check if we're at max concurrent jobs
        processing_count = await self.redis.scard(self.processing_key)
        if processing_count >= self.max_concurrent_jobs:
            return None
        
        # Get highest priority job (lowest score)
        result = await self.redis.zpopmin(self.queue_key, 1)
        if not result:
            return None
        
        job_id = result[0][0]
        
        # Add to processing set
        await self.redis.sadd(self.processing_key, job_id)
        
        return job_id

    async def _process_job(self, job_id: str):
        """Process a single batch job"""
        try:
            # Get job data
            job = await self.get_job_status(job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return
            
            # Update job status
            job.status = JobStatus.PROCESSING
            job.started_at = time.time()
            await self._update_job(job)
            
            logger.info(f"Processing job {job_id} with {len(job.prompts)} prompts")
            
            # Get provider
            if job.provider not in self.providers:
                raise ValueError(f"Provider {job.provider} not available")
            
            provider = self.providers[job.provider]
            
            # Process prompts in batches
            batch_size = job.parameters.get('batch_size', 5)
            results = []
            
            for i in range(0, len(job.prompts), batch_size):
                batch_prompts = job.prompts[i:i + batch_size]
                
                try:
                    # Generate images for this batch
                    if hasattr(provider, 'generate_batch'):
                        batch_results = await provider.generate_batch(batch_prompts, **job.parameters)
                    else:
                        # Fallback to individual generation
                        batch_results = []
                        for prompt in batch_prompts:
                            result = await provider.generate_image(prompt, **job.parameters)
                            batch_results.append(result)
                    
                    results.extend(batch_results)
                    
                    # Update progress
                    job.progress = min(len(results), job.total)
                    await self._update_job(job)
                    
                    # Small delay between batches
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error processing batch in job {job_id}: {e}")
                    # Add failed results for this batch
                    for prompt in batch_prompts:
                        results.append({
                            "success": False,
                            "error": str(e),
                            "prompt": prompt,
                            "provider": job.provider
                        })
            
            # Job completed
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.results = results
            job.progress = job.total
            
            await self._update_job(job)
            await self._update_stats("completed", 1)
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            
            # Update job as failed
            job = await self.get_job_status(job_id)
            if job:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                job.error = str(e)
                job.retry_count += 1
                
                # Retry if under max retries
                if job.retry_count < job.max_retries:
                    job.status = JobStatus.QUEUED
                    job.completed_at = None
                    job.error = None
                    
                    # Re-queue with lower priority
                    await self.redis.zadd(
                        self.queue_key,
                        {job_id: -(job.priority - job.retry_count)}
                    )
                    logger.info(f"Job {job_id} requeued for retry {job.retry_count}")
                
                await self._update_job(job)
                await self._update_stats("failed", 1)
        
        finally:
            # Remove from processing set
            await self.redis.srem(self.processing_key, job_id)

    async def _update_job(self, job: BatchJob):
        """Update job data in Redis"""
        await self.redis.hset(
            self.jobs_key,
            job.job_id,
            json.dumps(job.to_dict())
        )

    async def _update_stats(self, stat_name: str, increment: int = 1):
        """Update processing statistics"""
        await self.redis.hincrby(self.stats_key, stat_name, increment)