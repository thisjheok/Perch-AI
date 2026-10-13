export interface DownloadProgress {
    loaded: number;
    total?: number;
}
export declare function downloadModel(url: string, onProgress?: (progress: DownloadProgress) => void): Promise<ArrayBuffer>;
//# sourceMappingURL=downloader.d.ts.map