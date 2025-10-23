"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import Toast from "@/components/Toast";
import Loading from "@/components/Loading";

export default function NewCampaign() {
    const router = useRouter();
    const [campaignName, setCampaignName] = useState("");
    const [campaignDescription, setCampaignDescription] = useState("");
    const [image, setImage] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [toast, setToast] = useState<{
        type: "success" | "error";
        message: string;
    } | null>(null);
    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const [loading, setLoading] = useState(false);

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    // Auto-hide toast
    useEffect(() => {
        if (toast) {
            const timer = setTimeout(() => setToast(null), 3000);
            return () => clearTimeout(timer);
        }
    }, [toast]);

    // Show preview before upload
    const handleImageChange = (file: File) => {
        setImage(file);
        const reader = new FileReader();
        reader.onloadend = () => setPreview(reader.result as string);
        reader.readAsDataURL(file);
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!campaignName.trim() || !campaignDescription.trim()) {
            setToast({
                type: "error",
                message: "Name and description are required.",
            });
            return;
        }

        setLoading(true);
        try {
            const formData = new FormData();
            formData.append("campaign_name", campaignName);
            formData.append("campaign_description", campaignDescription);
            if (image) formData.append("campaign_image", image);

            const res = await fetch(`${baseUrl}/campaign/`, {
                method: "POST",
                body: formData,
            });

            const data = await res.json();
            if (!res.ok)
                throw new Error(data.error || "Failed to create campaign");

            setToast({
                type: "success",
                message: `Campaign "${campaignName}" created!`,
            });

            // Reset form
            setCampaignName("");
            setCampaignDescription("");
            setImage(null);
            setPreview(null);

            // Navigate to campaign selection
            setTimeout(() => router.push("/#campaign_selection"), 500);
        } catch (err: any) {
            setToast({
                type: "error",
                message: err.message || "Unexpected error occurred.",
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="w-screen h-screen bg-purple-colour flex items-center justify-center relative">
            {loading && <Loading />}
            <form
                onSubmit={handleSubmit}
                className="flex flex-col gap-2 p-8 max-w-2xl w-full bg-white rounded-xl shadow-lg relative border-2 border-purple"
            >
                {/* X button */}
                <button
                    type="button"
                    onClick={() => router.push("/")}
                    className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 text-3xl font-bold"
                >
                    ×
                </button>

                <h1 className="text-4xl text-center pb-4 purple-colour font-bold">
                    Create a Campaign
                </h1>

                {/* Name input */}
                <label className="block text-lg purple-colour font-semibold mb-1">
                    Name Your Campaign
                </label>
                <p className="text-sm text-gray-900 mb-1">
                    This will name your campaign.
                </p>
                <input
                    type="text"
                    placeholder="e.g. The Wild Beyond Witchlight"
                    className="border p-2 rounded w-full mb-3 text-black"
                    value={campaignName}
                    onChange={(e) => setCampaignName(e.target.value)}
                    required
                />

                {/* Description */}
                <label className="block text-lg purple-colour font-semibold mb-1">
                    Campaign Description
                </label>
                <p className="text-sm text-gray-900 mb-1">
                    Provide a short description of your campaign.
                </p>
                <textarea
                    placeholder="e.g. My first D&D game."
                    className="border p-2 rounded w-full h-24 mb-3 text-black"
                    value={campaignDescription}
                    onChange={(e) => setCampaignDescription(e.target.value)}
                />

                {/* Image Upload */}
                <label className="block text-lg purple-colour font-semibold mb-1">
                    Upload Campaign Image
                </label>
                <p className="text-sm text-gray-900 mb-1">
                    Choose an image that represents your campaign.
                </p>
                <div className="flex flex-col gap-2 items-center">
                    {/* Clickable area */}
                    <div
                        onClick={() => fileInputRef.current?.click()}
                        className="w-64 h-64 border-2 border-dashed border-gray-700 rounded-xl flex items-center justify-center cursor-pointer overflow-hidden bg-gray-300 relative"
                    >
                        {preview ? (
                            <img
                                src={preview}
                                alt="Preview"
                                className="w-full h-full object-cover opacity-80"
                            />
                        ) : (
                            <span
                                className="absolute text-white text-center px-2"
                                style={{
                                    textShadow: `
          1px 1px 0 #353434ff,
          -1px 1px 0 #353434ff,
          1px -1px 0 #353434ff,
          -1px -1px 0 #353434ff,
          0 1px 0 #353434ff,
          0 -1px 0 #353434ff,
          1px 0 0 #353434ff,
          -1px 0 0 #353434ff
        `,
                                }}
                            >
                                Click to pick an image
                            </span>
                        )}
                        {/* Remove image button */}
                        {preview && (
                            <button
                                type="button"
                                className="absolute top-2 right-2 w-6 h-6 flex items-center justify-center bg-gray-200 text-white font-bold rounded-full text-sm hover:bg-gray-400 cursor-pointer"
                                onClick={() => {
                                    setImage(null);
                                    setPreview(null);
                                    if (fileInputRef.current)
                                        fileInputRef.current.value = "";
                                }}
                            >
                                <img
                                    src="/svg/x-circle.svg"
                                    className="w-6 h-6" // adjust size as needed
                                />
                            </button>
                        )}
                    </div>

                    {/* Hidden file input */}
                    <input
                        type="file"
                        accept="image/*"
                        className="hidden"
                        ref={fileInputRef}
                        onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (!file) return;
                            handleImageChange(file);
                        }}
                    />
                </div>
                <div className="flex gap-2 justify-end">
                    <button
                        type="submit"
                        disabled={loading}
                        className="px-3 py-1 bg-green-600 rounded hover:bg-green-700 font-bold cursor-pointer"
                    >
                        {loading ? "Creating..." : "Create"}
                    </button>
                    <button
                        onClick={() => router.push("/")}
                        className="px-3 py-1 bg-gray-500 rounded hover:bg-gray-600 font-bold cursor-pointer"
                    >
                        Cancel
                    </button>
                </div>
            </form>
        </div>
    );
}
