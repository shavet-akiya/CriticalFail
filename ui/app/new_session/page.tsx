"use client"
import { useRef, useState } from "react";
import Link from "next/link";
import { Upload, Mic, Clock, CheckCircle, XCircle, Loader2 } from 'lucide-react';

export default function NewSession() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [uploadError, setUploadError] = useState<string>("");

  const handleUploadClick = () => {
    if (!isUploading) {
      fileInputRef.current?.click();
    }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    console.log("Selected file:", file.name);
    setIsUploading(true);
    setUploadStatus("Uploading audio file...");
    setUploadError("");
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('min_speakers', '2');
      formData.append('max_speakers', '8');
      
      // Submit job
      const response = await fetch('/api/speech/upload', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Upload failed');
      }
      
      const result = await response.json();
      console.log('Job submitted:', result.job_id);
      
      if (!result.job_id) {
        throw new Error('No job ID returned');
      }
      
      // Poll for completion
      setUploadStatus(`Processing audio (Job: ${result.job_id.substring(0, 8)}...)`);
      
      let attempts = 0;
      const maxAttempts = 400; // 20 minutes
      
      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 seconds
        attempts++;
        
        const statusResponse = await fetch(`/api/speech/status/${result.job_id}`);
        if (!statusResponse.ok) {
          console.error('Status check failed');
          continue;
        }
        
        const statusData = await statusResponse.json();
        console.log('Job status:', statusData.status);
        
        if (statusData.status === 'processing') {
          setUploadStatus(`Processing audio... (${Math.floor(attempts * 3 / 60)}m ${(attempts * 3) % 60}s)`);
        }
        
        if (statusData.status === 'completed' && statusData.result) {
          // Don't auto-download - just show success message
          setUploadStatus(`✅ Success! Transcript saved with ${statusData.result.speaker_count || 0} speakers identified.`);
          
          setTimeout(() => setUploadStatus(""), 10000); // Clear after 10 seconds
          break;
        }
        
        if (statusData.status === 'failed') {
          throw new Error(statusData.error || 'Processing failed');
        }
      }
      
      if (attempts >= maxAttempts) {
        throw new Error('Processing timeout');
      }
      
    } catch (error: any) {
      console.error('Error:', error);
      setUploadError(error.message || 'Failed to process audio file');
      setUploadStatus("");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh] gap-12 p-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-5xl font-bold text-gray-800 mb-4">Create New Session</h1>
        <p className="text-lg text-gray-600">Upload audio or start recording your D&D session</p>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-6">
        <button
          className="flex items-center gap-3 px-8 py-4 bg-indigo-600 text-white text-lg font-semibold rounded-lg hover:bg-indigo-700 transition-colors shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={handleUploadClick}
          disabled={isUploading}>
          {isUploading ? (
            <>
              <Loader2 className="animate-spin" size={24} />
              Processing...
            </>
          ) : (
            <>
              <Upload size={24} />
              Upload Audio File
            </>
          )}
        </button>
        
        <Link href="/new_session/recording">
          <button 
            className="flex items-center gap-3 px-8 py-4 bg-red-500 text-white text-lg font-semibold rounded-lg hover:bg-red-600 transition-colors shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
            disabled={isUploading}
          >
            <Mic size={24} />
            Start Recording
          </button>
        </Link>
      </div>

      {/* Status Messages */}
      {uploadStatus && (
        <div className={`flex items-center gap-3 p-4 rounded-lg shadow ${
          uploadStatus.includes('✅') 
            ? 'bg-green-50 border-l-4 border-green-500' 
            : 'bg-blue-50 border-l-4 border-blue-500'
        }`}>
          {uploadStatus.includes('✅') ? (
            <CheckCircle className="text-green-600" size={20} />
          ) : (
            <Loader2 className={isUploading ? "animate-spin text-blue-600" : "text-blue-600"} size={20} />
          )}
          <p className={uploadStatus.includes('✅') ? "text-green-800 font-medium" : "text-blue-800 font-medium"}>
            {uploadStatus}
          </p>
        </div>
      )}

      {uploadError && (
        <div className="flex items-center gap-3 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg shadow">
          <XCircle className="text-red-600" size={20} />
          <p className="text-red-800 font-medium">{uploadError}</p>
        </div>
      )}

      {/* Info Card */}
      <div className="max-w-2xl mt-8 p-6 bg-gray-50 rounded-lg shadow">
        <h3 className="text-xl font-semibold text-gray-800 mb-3">How it works:</h3>
        <ul className="space-y-2 text-gray-700">
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 font-bold">1.</span>
            <span>Upload an audio file or record your session live</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 font-bold">2.</span>
            <span>AI transcribes the audio and identifies different speakers</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 font-bold">3.</span>
            <span>Transcript is automatically saved in the transcripts folder</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-indigo-600 font-bold">4.</span>
            <span>Large files (1+ hour) may take 15-20 minutes to process</span>
          </li>
        </ul>
      </div>

      <input
        type="file"
        accept="audio/*"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}