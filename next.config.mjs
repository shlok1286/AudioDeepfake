/** @type {import('next').NextConfig} */
const backendUrl = (process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000').trim().replace(/\/$/, '')

const nextConfig = {
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/api/:path*`,
      },
    ]
  },
}

export default nextConfig
