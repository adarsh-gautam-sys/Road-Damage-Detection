/** Target size that fits within `maxEdge` on the long side. Never upscales. */
export function computeResizeDims(w: number, h: number, maxEdge: number): { w: number; h: number } {
  const longest = Math.max(w, h);
  if (longest <= maxEdge) return { w, h };
  const k = maxEdge / longest;
  return { w: Math.round(w * k), h: Math.round(h * k) };
}

/** Draw a bitmap/video frame to a canvas at a bounded size and return a JPEG blob. */
export async function toJpegBlob(
  source: CanvasImageSource & { width?: number; height?: number; videoWidth?: number; videoHeight?: number },
  maxEdge: number,
  quality: number,
): Promise<{ blob: Blob; width: number; height: number }> {
  const sw = source.videoWidth ?? source.width ?? 0;
  const sh = source.videoHeight ?? source.height ?? 0;
  if (!sw || !sh) throw new Error("Image source has no dimensions yet.");
  const { w, h } = computeResizeDims(sw, sh, maxEdge);
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable.");
  ctx.drawImage(source, 0, 0, w, h);
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
  if (!blob) throw new Error("Could not encode the frame as JPEG.");
  return { blob, width: w, height: h };
}

/** Decode a user-picked File to a drawable bitmap (honours EXIF orientation). */
export async function fileToBitmap(file: File): Promise<ImageBitmap> {
  return createImageBitmap(file, { imageOrientation: "from-image" });
}
