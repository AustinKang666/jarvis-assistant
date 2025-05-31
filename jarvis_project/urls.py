"""
Django URL配置
"""
from django.urls import path
from jarvis_project import views

# API版本前綴
API_PREFIX = "api/v1"

urlpatterns = [
    # 基礎端點
    path('', views.index, name='index'),
    
    # API端點
    path(f'{API_PREFIX}/jarvis/', views.jarvis_api, name='jarvis_api'),
    path(f'{API_PREFIX}/health/', views.health_check, name='health_check'),
    path(f'{API_PREFIX}/upload/', views.upload_file, name='upload_file'),
    path(f'{API_PREFIX}/rebuild_kb/', views.rebuild_knowledge_base, name='rebuild_kb'),
    
    # 快取相關端點
    path(f'{API_PREFIX}/cache_stats/', views.cache_stats, name='cache_stats'),
    path(f'{API_PREFIX}/clear_cache/', views.clear_cache, name='clear_cache'),
    
    # 內容過濾相關端點
    path(f'{API_PREFIX}/safety_config/', views.safety_config, name='safety_config'),
    path(f'{API_PREFIX}/test_safety_filter/', views.test_safety_filter, name='test_safety_filter'),
    
    # 語音相關端點
    path(f'{API_PREFIX}/speech_to_text/', views.speech_to_text, name='speech_to_text'),
    path(f'{API_PREFIX}/text_to_speech/', views.text_to_speech, name='text_to_speech'),

    # 視覺相關端點
    path(f'{API_PREFIX}/analyze_image/', views.analyze_image, name='analyze_image'),
    
    # 股票相關端點
    path(f'{API_PREFIX}/analyze_stock/', views.analyze_stock, name='analyze_stock'),
]