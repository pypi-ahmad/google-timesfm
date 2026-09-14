import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  agentRules: false,
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: "http://127.0.0.1:8001/api/v1/:path*",
      },
    ];
  },
};
export default nextConfig;
