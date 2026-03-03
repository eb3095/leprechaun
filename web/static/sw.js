/**
 * Leprechaun Trading Bot - Service Worker
 * Provides offline support and caching for the PWA
 */

const CACHE_NAME = 'leprechaun-v1';
const OFFLINE_URL = '/dashboard';

const STATIC_ASSETS = [
    '/dashboard',
    '/positions',
    '/trades',
    '/sentiment',
    '/alerts',
    '/settings',
    '/login',
    '/web/static/css/styles.css',
    '/web/static/js/main.js',
    '/web/static/manifest.json'
];

const CDN_ASSETS = [
    'https://cdn.tailwindcss.com',
    'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => self.skipWaiting())
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        if (cacheName !== CACHE_NAME) {
                            console.log('[SW] Deleting old cache:', cacheName);
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
            .then(() => self.clients.claim())
    );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip non-GET requests
    if (request.method !== 'GET') {
        return;
    }
    
    // Skip API calls - always go to network
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(
            fetch(request)
                .catch(() => new Response(
                    JSON.stringify({ error: 'offline', message: 'Network unavailable' }),
                    { headers: { 'Content-Type': 'application/json' }, status: 503 }
                ))
        );
        return;
    }
    
    // For HTML pages - network first, fallback to cache
    if (request.headers.get('Accept')?.includes('text/html')) {
        event.respondWith(
            fetch(request)
                .then((response) => {
                    const responseClone = response.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(request, responseClone);
                    });
                    return response;
                })
                .catch(() => {
                    return caches.match(request)
                        .then((response) => response || caches.match(OFFLINE_URL));
                })
        );
        return;
    }
    
    // For static assets - cache first, fallback to network
    event.respondWith(
        caches.match(request)
            .then((response) => {
                if (response) {
                    return response;
                }
                
                return fetch(request)
                    .then((response) => {
                        // Don't cache non-successful responses
                        if (!response || response.status !== 200) {
                            return response;
                        }
                        
                        // Clone the response
                        const responseClone = response.clone();
                        
                        // Cache the response
                        caches.open(CACHE_NAME).then((cache) => {
                            cache.put(request, responseClone);
                        });
                        
                        return response;
                    });
            })
    );
});

// Handle push notifications
self.addEventListener('push', (event) => {
    const options = {
        body: event.data?.text() || 'New notification from Leprechaun',
        icon: '/web/static/icons/icon-192.png',
        badge: '/web/static/icons/icon-72.png',
        vibrate: [100, 50, 100],
        data: {
            dateOfArrival: Date.now(),
            primaryKey: 1
        },
        actions: [
            { action: 'view', title: 'View Dashboard' },
            { action: 'close', title: 'Dismiss' }
        ]
    };
    
    event.waitUntil(
        self.registration.showNotification('Leprechaun Trading Bot', options)
    );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    
    if (event.action === 'view' || !event.action) {
        event.waitUntil(
            clients.openWindow('/dashboard')
        );
    }
});

// Background sync for offline actions
self.addEventListener('sync', (event) => {
    if (event.tag === 'sync-trades') {
        event.waitUntil(syncTrades());
    }
});

async function syncTrades() {
    console.log('[SW] Syncing trades...');
}

console.log('[SW] Service worker loaded');
