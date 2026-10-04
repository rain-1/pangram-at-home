interface Env { BUCKET: R2Bucket; TOKEN_SHA256: string; EXPIRES_AT: string }
export default { async fetch(request: Request, env: Env): Promise<Response> {
  if (Date.now() > Number(env.EXPIRES_AT)*1000) return new Response('Expired',{status:403});
  const hash=await crypto.subtle.digest('SHA-256',new TextEncoder().encode((request.headers.get('Authorization')||'').replace(/^Bearer /,'')));
  if([...new Uint8Array(hash)].map(x=>x.toString(16).padStart(2,'0')).join('')!==env.TOKEN_SHA256) return new Response('Unauthorized',{status:401});
  const key=new URL(request.url).pathname.slice(1);
  if(!/^atlas-public\/(objects\/[a-f0-9]{64}\.json\.gz|catalogue\.json)$/.test(key))return new Response('Not found',{status:404});
  if(request.method==='PUT'){
    const object=await env.BUCKET.put(key,request.body,{onlyIf:request.headers,httpMetadata:{contentType:key.endsWith('.gz')?'application/octet-stream':'application/json'}});
    if(!object)return new Response('Publication changed; retry from the current catalogue',{status:412});
    return Response.json({size:object.size,etag:object.etag});
  }
  if(request.method==='GET'){
    const object=await env.BUCKET.get(key);
    return object?new Response(object.body,{headers:{ETag:object.httpEtag}}):new Response('Not found',{status:404});
  }
  return new Response('Method not allowed',{status:405});
}} satisfies ExportedHandler<Env>;
