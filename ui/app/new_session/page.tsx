"use client"
import { useRef } from "react";

export default function NewSession() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      console.log("Selected file:", file.name);
    }
  };

  return (
    // 
    <div className="flex flex-col items-center justify-center gap-16">
      <img
        src="/svg/recording_page_dice.svg"
        alt="D20"
        className="w-1/2 sm:w-1/3 md:w-1/2 h-auto"
      />

      <div className="flex gap-4">
        <button
          className="btn btn-outline rounded-md"
          onClick={handleUploadClick}>
          Upload Session</button>
        <button className="btn btn-outline rounded-md">Start Recording</button>
      </div>

      <input
        type="file"
        accept=".mp3"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
      />
    </div>
  );
}
