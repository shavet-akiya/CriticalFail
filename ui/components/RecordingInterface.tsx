import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Mic, MicOff, Pause, Play, Download, Upload, Loader2, Users, Trash2 } from 'lucide-react';

interface TranscriptData {
  success: boolean;
  transcript: string;
  file_path?: string;
  error?: string;
  speakers?: string[];
  speaker_count?: number;
  metadata?: {
    processed_at: string;
    min_speakers: number;
    max_speakers: number;
  };
  file_info?: {
    original_name: string;
    saved_name: string;
    size_mb: number;
  };
  structured_data?: any;
  ai_error?: string;
}

const RecordingInterface: React.FC = () => {
  // Recording states
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  
  // Audio data
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  
  // Results
  const [transcript, setTranscript] = useState<TranscriptData | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  // Settings
  const [minSpeakers, setMinSpeakers] = useState(2);
  const [maxSpeakers, setMaxSpeakers] = useState(8);
  
  // Refs
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  // Timer for recording duration
  useEffect(() => {
    if (isRecording && !isPaused) {
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [isRecording, isPaused]);

  const formatTime = (seconds: number): string => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    return `${hrs.toString().padStart(2, '0')}:${mins
      .toString()
      .padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const startRecording = async () => {
    try {
      setError(null);
      
      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        } 
      });
      
      streamRef.current = stream;

      // Determine best mime type for the browser
      let mimeType = 'audio/webm';
      const types = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/ogg;codecs=opus',
        'audio/mp4',
        'audio/mpeg'
      ];
      
      for (const type of types) {
        if (MediaRecorder.isTypeSupported(type)) {
          mimeType = type;
          break;
        }
      }

      console.log('Using mime type:', mimeType);

      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
        setAudioBlob(audioBlob);
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        console.log('Recording saved, size:', (audioBlob.size / 1024 / 1024).toFixed(2), 'MB');
      };

      mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        setError('Recording error occurred');
      };

      // Start recording
      mediaRecorder.start(1000); // Collect data every second
      setIsRecording(true);
      setRecordingTime(0);
      setTranscript(null);
      console.log('Recording started');
      
    } catch (err) {
      console.error('Error starting recording:', err);
      setError('Failed to start recording. Please ensure microphone permissions are granted.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsPaused(false);
      
      // Stop all tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
      console.log('Recording stopped');
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        setIsPaused(false);
        console.log('Recording resumed');
      } else {
        mediaRecorderRef.current.pause();
        setIsPaused(true);
        console.log('Recording paused');
      }
    }
  };

  const uploadAudio = async () => {
    if (!audioBlob) {
      setError('No audio to upload');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const formData = new FormData();
      
      // Create a proper file name with extension
      const fileName = `recording_${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
      
      // Append the audio blob as a file
      formData.append('file', audioBlob, fileName);
      formData.append('min_speakers', minSpeakers.toString());
      formData.append('max_speakers', maxSpeakers.toString());

      console.log('Uploading audio:', fileName, 'Size:', (audioBlob.size / 1024 / 1024).toFixed(2), 'MB');

      const response = await fetch('http://localhost:8000/api/speech/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      console.log('Processing complete:', data);
      setTranscript(data);
      
      if (!data.success && data.error) {
        setError(data.error);
      }
      
    } catch (err) {
      console.error('Upload error:', err);
      setError(err instanceof Error ? err.message : 'Failed to upload and process audio');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setError(null);
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('min_speakers', minSpeakers.toString());
      formData.append('max_speakers', maxSpeakers.toString());

      console.log('Uploading file:', file.name, 'Size:', (file.size / 1024 / 1024).toFixed(2), 'MB');

      const response = await fetch('http://localhost:8000/api/speech/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || errorData.detail || 'Upload failed');
      }

      const data = await response.json();
      console.log('Processing complete:', data);
      setTranscript(data);
      
      if (!data.success && data.error) {
        setError(data.error);
      }

    } catch (err) {
      console.error('Upload error:', err);
      setError(err instanceof Error ? err.message : 'Failed to process audio file');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadAudio = () => {
    if (audioUrl && audioBlob) {
      const a = document.createElement('a');
      a.href = audioUrl;
      a.download = `recording_${new Date().toISOString().replace(/[:.]/g, '-')}.webm`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const downloadTranscript = () => {
    if (transcript) {
      const dataStr = JSON.stringify(transcript, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
      
      const a = document.createElement('a');
      a.href = dataUri;
      a.download = `transcript_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const clearRecording = () => {
    setAudioBlob(null);
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }
    setRecordingTime(0);
    setTranscript(null);
    setError(null);
    audioChunksRef.current = [];
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-3xl font-bold mb-6 text-gray-800">D&D Session Recorder</h2>
        
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
            <p className="font-medium">Error</p>
            <p className="text-sm">{error}</p>
          </div>
        )}

        {/* Recording Controls */}
        <div className="mb-6 p-6 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-4">
              {!isRecording ? (
                <button
                  onClick={startRecording}
                  className="flex items-center px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors shadow-md"
                  disabled={isProcessing}
                >
                  <Mic className="mr-2" size={20} />
                  Start Recording
                </button>
              ) : (
                <>
                  <button
                    onClick={stopRecording}
                    className="flex items-center px-6 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors shadow-md"
                  >
                    <MicOff className="mr-2" size={20} />
                    Stop
                  </button>
                  <button
                    onClick={pauseRecording}
                    className="flex items-center px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors shadow-md"
                  >
                    {isPaused ? (
                      <>
                        <Play className="mr-2" size={20} />
                        Resume
                      </>
                    ) : (
                      <>
                        <Pause className="mr-2" size={20} />
                        Pause
                      </>
                    )}
                  </button>
                </>
              )}
              
              {/* File Upload */}
              <div className="relative">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                  disabled={isRecording || isProcessing}
                />
                <label
                  htmlFor="file-upload"
                  className={`flex items-center px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors shadow-md cursor-pointer ${
                    (isRecording || isProcessing) ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  <Upload className="mr-2" size={20} />
                  Upload Audio
                </label>
              </div>
            </div>
            
            <div className="text-2xl font-mono font-bold text-gray-700">
              {formatTime(recordingTime)}
            </div>
          </div>

          {/* Recording Status Indicator */}
          {isRecording && (
            <div className="flex items-center bg-white/50 rounded-lg px-4 py-2 inline-flex">
              <div className={`w-3 h-3 rounded-full mr-3 ${
                isPaused ? 'bg-yellow-500' : 'bg-red-500 animate-pulse'
              }`} />
              <span className="text-sm font-medium text-gray-700">
                {isPaused ? 'Recording Paused' : 'Recording in Progress...'}
              </span>
            </div>
          )}
        </div>

        {/* Speaker Settings */}
        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center mb-3">
            <Users className="mr-2 text-indigo-600" size={20} />
            <h3 className="font-semibold text-gray-800">Speaker Detection Settings</h3>
          </div>
          <div className="flex space-x-6">
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">
                Minimum Speakers
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={minSpeakers}
                onChange={(e) => setMinSpeakers(parseInt(e.target.value) || 2)}
                className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                disabled={isRecording || isProcessing}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-600 mb-1">
                Maximum Speakers
              </label>
              <input
                type="number"
                min="2"
                max="20"
                value={maxSpeakers}
                onChange={(e) => setMaxSpeakers(parseInt(e.target.value) || 8)}
                className="w-24 px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                disabled={isRecording || isProcessing}
              />
            </div>
          </div>
        </div>

        {/* Audio Preview and Actions */}
        {audioUrl && !isRecording && (
          <div className="mb-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="font-semibold mb-3 text-gray-800">Recorded Audio</h3>
            <audio controls className="w-full mb-4">
              <source src={audioUrl} />
              Your browser does not support the audio element.
            </audio>
            
            <div className="flex space-x-3">
              <button
                onClick={uploadAudio}
                disabled={isProcessing}
                className="flex items-center px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isProcessing ? (
                  <>
                    <Loader2 className="mr-2 animate-spin" size={18} />
                    Processing...
                  </>
                ) : (
                  <>
                    <Upload className="mr-2" size={18} />
                    Process Audio
                  </>
                )}
              </button>
              
              <button
                onClick={downloadAudio}
                className="flex items-center px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors"
              >
                <Download className="mr-2" size={18} />
                Download Audio
              </button>
              
              <button
                onClick={clearRecording}
                className="flex items-center px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
              >
                <Trash2 className="mr-2" size={18} />
                Clear
              </button>
            </div>
          </div>
        )}

        {/* Processing Indicator */}
        {isProcessing && (
          <div className="mb-6 p-8 bg-indigo-50 rounded-lg text-center">
            <Loader2 className="mx-auto mb-4 animate-spin text-indigo-600" size={48} />
            <p className="text-lg font-medium text-gray-700">Processing audio with WhisperX...</p>
            <p className="text-sm text-gray-600 mt-2">This may take a few minutes for longer recordings</p>
          </div>
        )}

        {/* Transcript Display */}
        {transcript && !isProcessing && (
          <div className="p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-semibold text-gray-800">Transcription Results</h3>
              <button
                onClick={downloadTranscript}
                className="flex items-center px-4 py-2 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors"
              >
                <Download className="mr-2" size={16} />
                Download JSON
              </button>
            </div>
            
            {/* Success/Error Status */}
            {transcript.success ? (
              <div className="mb-4 p-3 bg-green-100 border border-green-300 rounded-lg">
                <p className="text-green-800 font-medium">✓ Transcription Successful</p>
              </div>
            ) : (
              <div className="mb-4 p-3 bg-red-100 border border-red-300 rounded-lg">
                <p className="text-red-800 font-medium">✗ Transcription Failed</p>
                {transcript.error && (
                  <p className="text-red-700 text-sm mt-1">{transcript.error}</p>
                )}
              </div>
            )}
            
            {/* Metadata */}
            <div className="mb-4 bg-white p-4 rounded-lg">
              <h4 className="font-medium text-gray-700 mb-2">Processing Information</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                {transcript.speaker_count !== undefined && (
                  <div>
                    <span className="text-gray-600">Speakers Detected:</span>
                    <span className="ml-2 font-medium">{transcript.speaker_count}</span>
                  </div>
                )}
                {transcript.speakers && transcript.speakers.length > 0 && (
                  <div>
                    <span className="text-gray-600">Speaker IDs:</span>
                    <span className="ml-2 font-medium">{transcript.speakers.join(', ')}</span>
                  </div>
                )}
                {transcript.file_info && (
                  <>
                    <div>
                      <span className="text-gray-600">File Size:</span>
                      <span className="ml-2 font-medium">{transcript.file_info.size_mb.toFixed(2)} MB</span>
                    </div>
                    <div>
                      <span className="text-gray-600">Saved As:</span>
                      <span className="ml-2 font-medium text-xs">{transcript.file_info.saved_name}</span>
                    </div>
                  </>
                )}
                {transcript.metadata?.processed_at && (
                  <div className="col-span-2">
                    <span className="text-gray-600">Processed At:</span>
                    <span className="ml-2 font-medium">
                      {new Date(transcript.metadata.processed_at).toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            </div>
            
            {/* Transcript Text */}
            {transcript.transcript && (
              <div className="mb-4">
                <h4 className="font-medium text-gray-700 mb-2">Transcript</h4>
                <div className="bg-white p-4 rounded-lg border border-gray-200 max-h-96 overflow-y-auto">
                  <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700">
                    {transcript.transcript}
                  </pre>
                </div>
              </div>
            )}
            
            {/* AI Structured Data */}
            {transcript.structured_data && (
              <div className="mb-4">
                <h4 className="font-medium text-gray-700 mb-2">AI-Extracted Game Data</h4>
                <div className="bg-white p-4 rounded-lg border border-gray-200 max-h-64 overflow-y-auto">
                  <pre className="text-xs">
                    {JSON.stringify(transcript.structured_data, null, 2)}
                  </pre>
                </div>
              </div>
            )}
            
            {/* AI Error */}
            {transcript.ai_error && (
              <div className="p-3 bg-yellow-100 border border-yellow-300 rounded-lg">
                <p className="text-yellow-800 text-sm">
                  <strong>AI Processing Note:</strong> {transcript.ai_error}
                </p>
              </div>
            )}
            
            {/* File Path */}
            {transcript.file_path && (
              <div className="mt-4 text-xs text-gray-500">
                <span>Transcript saved to: {transcript.file_path}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default RecordingInterface;