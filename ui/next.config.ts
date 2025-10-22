import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  eslint: { ignoreDuringBuilds: true },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:9000/:path*",
      },
    ];
  },
  // Keep long HTTP requests alive
  serverRuntimeConfig: {
    httpServerTimeout: 0, // 0 = no timeout
  },
    images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "9000",
        pathname: "/campaign_images/**",
      },
    ],
  },
};

export default nextConfig;
