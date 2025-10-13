import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  eslint: {
    ignoreDuringBuilds: true,
  },
  experimental: {
    proxyTimeout: 7200000, // 2 hours (was 1 hour)
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://server:9000/:path*",
      },
    ];
  },
};

export default nextConfig;