import { useState } from 'react';
import type { EnvironmentEntry } from '../api/redteam';

/** Task previews are local assets with recorded provenance, never inferred from another task. */
export function taskPreviewUrl(value: unknown): string | null {
  if (typeof value !== 'string' || !/^\/task-previews\/[A-Za-z0-9][A-Za-z0-9._/-]*\.(?:png|jpe?g|webp|avif)$/i.test(value)) return null;
  return value.includes('//') || value.split('/').some(part => part === '..') ? null : value;
}

const SOURCE_LABEL = { episode_frame: '真实运行帧', task_asset: '任务资源图', illustration: '任务示意图' };

export default function TaskPreview({ name, preview, placeholder = '暂无任务预览' }: {
  name: string; preview?: EnvironmentEntry['preview']; placeholder?: string;
}) {
  const [failedUrl, setFailedUrl] = useState('');
  const [loadedUrl, setLoadedUrl] = useState('');
  const url = taskPreviewUrl(preview?.url);
  const failed = !!url && failedUrl === url;
  const source = preview ? SOURCE_LABEL[preview.kind] : '';
  const status = preview && !url ? '预览地址无效' : failed ? '预览加载失败' : url ? '加载预览…' : placeholder;
  return <span className="task-preview">
    <span className="task-preview-image">
      {(!url || failed || loadedUrl !== url) && <span className="task-preview-placeholder" aria-hidden={!!url && !failed}>{status}</span>}
      {url && !failed && <img key={url} src={url} alt={`${name} · ${source}；${preview?.caption ?? ''}`}
        width={128} height={96} loading="lazy" decoding="async" className={loadedUrl === url ? 'loaded' : ''}
        onLoad={() => setLoadedUrl(url)} onError={() => setFailedUrl(url)} />}
    </span>
    {source && <span className="task-preview-source">{source}</span>}
  </span>;
}
