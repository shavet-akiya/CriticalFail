import { NextRequest } from 'next/server';

export const dynamic = 'force-dynamic';

export async function POST(request: NextRequest) {
  console.log('');
  console.log('='.repeat(80));
  console.log('[UPLOAD API] 🚀 NEW REQUEST RECEIVED');
  console.log('='.repeat(80));
  
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    console.log('[UPLOAD API] Step 1: Got file from browser');
    console.log('[UPLOAD API]   - Name:', file?.name);
    console.log('[UPLOAD API]   - Size:', (file?.size / 1024 / 1024).toFixed(2), 'MB');
    console.log('[UPLOAD API]   - Type:', file?.type);
    
    console.log('[UPLOAD API] Step 2: Forwarding to server...');
    console.log('[UPLOAD API]   - Target: http://server:9000/speech/upload');
    
    const response = await fetch('http://server:9000/speech/upload', {
      method: 'POST',
      body: formData,
    });
    
    console.log('[UPLOAD API] Step 3: Got response from server');
    console.log('[UPLOAD API]   - Status:', response.status);
    console.log('[UPLOAD API]   - OK:', response.ok);
    
    if (!response.ok) {
      const errorData = await response.json();
      console.error('[UPLOAD API] ❌ Server returned error:', errorData);
      return Response.json(errorData, { status: response.status });
    }
    
    const jobData = await response.json();
    
    console.log('[UPLOAD API] Step 4: Parsed response JSON');
    console.log('[UPLOAD API]   - Full response:', JSON.stringify(jobData, null, 2));
    console.log('[UPLOAD API]   - Has job_id?', !!jobData.job_id);
    console.log('[UPLOAD API]   - job_id value:', jobData.job_id);
    
    console.log('[UPLOAD API] Step 5: Returning response to client');
    console.log('[UPLOAD API] ✅ DONE - Client should now start polling');
    console.log('='.repeat(80));
    console.log('');
    
    // CRITICAL: Return immediately, let client poll
    return Response.json(jobData);
    
  } catch (error: any) {
    console.error('');
    console.error('='.repeat(80));
    console.error('[UPLOAD API] ❌ EXCEPTION CAUGHT');
    console.error('[UPLOAD API] Error type:', error.constructor.name);
    console.error('[UPLOAD API] Error message:', error.message);
    console.error('[UPLOAD API] Error stack:', error.stack);
    console.error('='.repeat(80));
    console.error('');
    
    return Response.json({ 
      error: error.message, 
      success: false 
    }, { status: 500 });
  }
}