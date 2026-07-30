/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Emits a self-contained server bundle, which keeps the dashboard image small.
  output: "standalone",
};

export default nextConfig;
