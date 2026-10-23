/* ProFace service worker — cache static assets for offline resilience */
const CACHE = 'proface-v1';
const ASSETS = [
  '/static/manifest.json',
  'https://cdn.tailwindcss.com?plugins=forms,typography',
  'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(ASSETS)).catch(() => {}));
});

self.addEventListener('fetch', (e) => {
  // Cache-first for static assets, network-first for everything else
  if (e.request.url.includes('/static/') || e.request.url.includes('cdn.')) {
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request))
    );
  }
});
