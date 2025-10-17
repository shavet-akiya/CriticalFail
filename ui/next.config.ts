import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  eslint: { ignoreDuringBuilds: true },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://server:9000/:path*",
      },
    ];
  },
  // Keep long HTTP requests alive
  serverRuntimeConfig: {
    httpServerTimeout: 0, // 0 = no timeout
  },
};

export default nextConfig;
