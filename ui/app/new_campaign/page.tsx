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
            {loading && <Loading />} {/* <-- Shows the loading overlay */}
            {toast && <Toast type={toast.type} message={toast.message} />}
            <form
                onSubmit={handleSubmit}
                className="flex flex-col gap-4 p-8 max-w-lg w-full bg-white rounded-xl shadow-lg relative"
            >
                {/* X button */}
                <button
                    type="button"
                    onClick={() => router.push("/")}
                    className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 text-2xl font-bold"
                >
                    ×
                </button>

                <h1 className="text-4xl text-center pb-4 obsidian-colour">
                    Create a Campaign
                </h1>

                {/* Name input */}
                <label className="block text-lg purple-colour font-semibold mb-2">
                    Name Your Campaign
                </label>
                <input
                    type="text"
                    placeholder="e.g. The Wild Beyond Witchlight"
                    className="input input-neutral w-full"
                    value={campaignName}
                    onChange={(e) => setCampaignName(e.target.value)}
                    required
                />

                {/* Description */}
                <label className="block text-lg purple-colour font-semibold mb-2">
                    Campaign Description
                </label>
                <textarea
                    placeholder="e.g. My first D&D game."
                    className="textarea textarea-neutral w-full"
                    value={campaignDescription}
                    onChange={(e) => setCampaignDescription(e.target.value)}
                    required
                />

                {/* Image Upload */}
                <label className="block text-lg purple-colour font-semibold mb-2">
                    Upload Campaign Image
                </label>
                <div className="flex flex-col gap-2">
                    <input
                        type="file"
                        accept="image/*"
                        className="file-input w-full"
                        ref={fileInputRef}
                        onChange={(e) => {
                            const file = e.target.files?.[0];
                            if (!file) return;
                            handleImageChange(file);
                        }}
                    />
                    {preview && (
                        <>
                            <img
                                src={preview}
                                alt="Preview"
                                className="rounded-xl max-h-64 object-contain mx-auto border border-gray-300"
                            />
                            <button
                                type="button"
                                className="btn btn-outline btn-error w-full"
                                onClick={() => {
                                    setImage(null);
                                    setPreview(null);
                                    if (fileInputRef.current)
                                        fileInputRef.current.value = "";
                                }}
                            >
                                Remove Image
                            </button>
                        </>
                    )}
                </div>

                <button
                    type="submit"
                    className="btn btn-primary w-full"
                    disabled={loading}
                >
                    {loading ? "Creating..." : "Create"}
                </button>
            </form>
        </div>
    );
}
