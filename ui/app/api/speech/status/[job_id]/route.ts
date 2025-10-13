import { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ job_id: string }> }
) {
  try {
    // Await params in Next.js 15+
    const { job_id } = await params;
    
    const response = await fetch(`http://server:9000/speech/status/${job_id}`);
    const data = await response.json();
    return Response.json(data);
  } catch (error: any) {
    return Response.json({ error: error.message }, { status: 500 });
  }
}