/**
 * Leprechaun Trading Bot - Main JavaScript
 * Handles API calls, authentication, and common UI functionality
 */

// Configuration
const API_BASE_URL = '/api/v1';
const REFRESH_INTERVAL = 30000;

// State
let refreshTimer = null;
let isRefreshing = false;

/**
 * Initialize the application
 */
function initializeApp() {
    checkAuth();
    updateUserDisplay();
    updateBotStatus();
    updateMarketStatus();
    updateLastUpdated();
    
    // Setup refresh interval
    refreshTimer = setInterval(() => {
        updateBotStatus();
        updateMarketStatus();
        updateLastUpdated();
    }, REFRESH_INTERVAL);
}

/**
 * Check authentication status
 */
function checkAuth() {
    const token = localStorage.getItem('access_token');
    if (!token && !window.location.pathname.includes('/login')) {
        // Allow viewing dashboard without auth but with limited functionality
        console.log('Running in guest mode');
    }
}

/**
 * Get the authentication token
 */
function getAuthToken() {
    return localStorage.getItem('access_token');
}

/**
 * Make an authenticated API call
 */
async function apiCall(endpoint, options = {}) {
    const token = getAuthToken();
    
    const headers = {
        'Content-Type': 'application/json',
        ...options.headers
    };
    
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    
    try {
        const response = await fetch(endpoint.startsWith('/') ? endpoint : `${API_BASE_URL}/${endpoint}`, {
            ...options,
            headers
        });
        
        // Handle 401 - token expired
        if (response.status === 401) {
            const refreshed = await refreshToken();
            if (refreshed) {
                // Retry the request with new token
                headers['Authorization'] = `Bearer ${localStorage.getItem('access_token')}`;
                const retryResponse = await fetch(endpoint.startsWith('/') ? endpoint : `${API_BASE_URL}/${endpoint}`, {
                    ...options,
                    headers
                });
                return await handleResponse(retryResponse);
            } else {
                handleAuthError();
                return null;
            }
        }
        
        return await handleResponse(response);
    } catch (error) {
        console.error('API call error:', error);
        showToast('Network error. Please check your connection.', 'error');
        return null;
    }
}

/**
 * Handle API response
 */
async function handleResponse(response) {
    if (!response.ok) {
        const error = await response.json().catch(() => ({ message: 'Unknown error' }));
        console.error('API error:', error);
        
        if (response.status === 429) {
            showToast('Rate limit exceeded. Please slow down.', 'warning');
        } else if (response.status >= 500) {
            showToast('Server error. Please try again later.', 'error');
        }
        
        return null;
    }
    
    return response.json();
}

/**
 * Refresh the authentication token
 */
async function refreshToken() {
    const refreshTokenValue = localStorage.getItem('refresh_token');
    if (!refreshTokenValue) return false;
    
    try {
        const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${refreshTokenValue}`
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            localStorage.setItem('access_token', data.access_token);
            if (data.refresh_token) {
                localStorage.setItem('refresh_token', data.refresh_token);
            }
            return true;
        }
    } catch (error) {
        console.error('Token refresh error:', error);
    }
    
    return false;
}

/**
 * Handle authentication errors
 */
function handleAuthError() {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    
    if (!window.location.pathname.includes('/login')) {
        showToast('Session expired. Please log in again.', 'warning');
        setTimeout(() => {
            window.location.href = '/login';
        }, 1500);
    }
}

/**
 * Handle logout
 */
async function handleLogout() {
    const token = getAuthToken();
    
    if (token) {
        try {
            await apiCall('/api/v1/auth/logout', { method: 'POST' });
        } catch (error) {
            console.error('Logout error:', error);
        }
    }
    
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user');
    
    window.location.href = '/login';
}

/**
 * Update user display in sidebar
 */
function updateUserDisplay() {
    const userStr = localStorage.getItem('user');
    const usernameDisplay = document.getElementById('username-display');
    const userInitial = document.getElementById('user-initial');
    
    if (userStr && usernameDisplay && userInitial) {
        try {
            const user = JSON.parse(userStr);
            usernameDisplay.textContent = user.username || 'User';
            userInitial.textContent = (user.username || 'U')[0].toUpperCase();
        } catch (e) {
            usernameDisplay.textContent = 'Guest';
            userInitial.textContent = 'G';
        }
    }
}

/**
 * Update bot status indicator
 */
async function updateBotStatus() {
    const badge = document.getElementById('bot-status-badge');
    if (!badge) return;
    
    try {
        const data = await apiCall('/api/v1/trading/status');
        if (data) {
            const status = data.status || 'stopped';
            badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
            
            if (status === 'running') {
                badge.className = 'px-2 py-1 text-xs font-medium rounded-full bg-green-900/50 text-green-400';
            } else if (status === 'halted') {
                badge.className = 'px-2 py-1 text-xs font-medium rounded-full bg-yellow-900/50 text-yellow-400';
            } else {
                badge.className = 'px-2 py-1 text-xs font-medium rounded-full bg-red-900/50 text-red-400';
            }
        }
    } catch (error) {
        badge.textContent = 'Unknown';
        badge.className = 'px-2 py-1 text-xs font-medium rounded-full bg-slate-600 text-slate-300';
    }
}

/**
 * Update market status indicator
 */
function updateMarketStatus() {
    const dot = document.getElementById('market-status-dot');
    const text = document.getElementById('market-status-text');
    if (!dot || !text) return;
    
    const now = new Date();
    const nyTime = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const hour = nyTime.getHours();
    const minute = nyTime.getMinutes();
    const day = nyTime.getDay();
    
    // Market hours: 9:30 AM - 4:00 PM ET, Mon-Fri
    const timeValue = hour + minute / 60;
    const isWeekday = day >= 1 && day <= 5;
    const isMarketHours = timeValue >= 9.5 && timeValue < 16;
    const isPreMarket = timeValue >= 4 && timeValue < 9.5;
    const isAfterHours = timeValue >= 16 && timeValue < 20;
    
    if (isWeekday && isMarketHours) {
        dot.className = 'w-2 h-2 rounded-full bg-green-500 pulse-dot';
        text.textContent = 'Market Open';
    } else if (isWeekday && isPreMarket) {
        dot.className = 'w-2 h-2 rounded-full bg-yellow-500';
        text.textContent = 'Pre-Market';
    } else if (isWeekday && isAfterHours) {
        dot.className = 'w-2 h-2 rounded-full bg-yellow-500';
        text.textContent = 'After Hours';
    } else {
        dot.className = 'w-2 h-2 rounded-full bg-red-500';
        text.textContent = 'Market Closed';
    }
}

/**
 * Update last updated timestamp
 */
function updateLastUpdated() {
    const el = document.getElementById('last-updated');
    if (!el) return;
    
    const now = new Date();
    el.textContent = `Updated: ${now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
}

/**
 * Refresh page data
 */
function refreshData() {
    if (isRefreshing) return;
    
    isRefreshing = true;
    const icon = document.getElementById('refresh-icon');
    if (icon) {
        icon.classList.add('animate-spin');
    }
    
    // Trigger page-specific refresh if defined
    if (typeof loadDashboardData === 'function') {
        loadDashboardData();
    } else if (typeof loadPositions === 'function') {
        loadPositions();
    } else if (typeof loadTrades === 'function') {
        loadTrades();
    } else if (typeof loadAlerts === 'function') {
        loadAlerts();
    }
    
    updateBotStatus();
    updateMarketStatus();
    updateLastUpdated();
    
    setTimeout(() => {
        isRefreshing = false;
        if (icon) {
            icon.classList.remove('animate-spin');
        }
        showToast('Data refreshed', 'success');
    }, 500);
}

/**
 * Show toast notification
 */
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const colors = {
        success: 'bg-green-900/90 text-green-100 border-green-700',
        error: 'bg-red-900/90 text-red-100 border-red-700',
        warning: 'bg-yellow-900/90 text-yellow-100 border-yellow-700',
        info: 'bg-slate-700/90 text-slate-100 border-slate-600'
    };
    
    const icons = {
        success: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>',
        error: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>',
        warning: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"/></svg>',
        info: '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>'
    };
    
    const toast = document.createElement('div');
    toast.className = `flex items-center p-4 rounded-lg shadow-lg border ${colors[type] || colors.info} toast-enter`;
    toast.innerHTML = `
        <span class="mr-3">${icons[type] || icons.info}</span>
        <span class="flex-1">${message}</span>
        <button onclick="this.parentElement.remove()" class="ml-4 opacity-70 hover:opacity-100">
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
            </svg>
        </button>
    `;
    
    container.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.classList.remove('toast-enter');
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, 5000);
}

/**
 * Format currency
 */
function formatCurrency(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) return '$--';
    
    const num = parseFloat(value);
    const absNum = Math.abs(num);
    
    if (absNum >= 1000000) {
        return '$' + (num / 1000000).toFixed(2) + 'M';
    } else if (absNum >= 1000) {
        return '$' + (num / 1000).toFixed(2) + 'K';
    }
    
    return '$' + num.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Format date
 */
function formatDate(dateString) {
    if (!dateString) return '--';
    
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}

/**
 * Format relative time
 */
function formatTimeAgo(timestamp) {
    if (!timestamp) return '--';
    
    const date = new Date(timestamp);
    const now = new Date();
    const seconds = Math.floor((now - date) / 1000);
    
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
    
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Format percentage
 */
function formatPercent(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) return '--%';
    return parseFloat(value).toFixed(decimals) + '%';
}

/**
 * Debounce function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Register service worker for PWA
 */
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/web/static/sw.js')
            .then(registration => {
                console.log('SW registered:', registration.scope);
            })
            .catch(error => {
                console.log('SW registration failed:', error);
            });
    });
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        apiCall,
        showToast,
        formatCurrency,
        formatDate,
        formatTimeAgo,
        formatPercent
    };
}
