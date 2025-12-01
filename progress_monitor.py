"""
Progress Monitor - утилита для мониторинга прогресса Phase 2
"""

import json
import time
from pathlib import Path
from datetime import datetime

def monitor_progress():
    """Monitor Phase 2 progress"""
    
    print("=== Phase 2 Progress Monitor ===")
    
    # Check if Phase 2 result exists
    result_file = Path('output/news_phase2_vllm_analysis.json')
    
    if result_file.exists():
        print(f"✓ Phase 2 result file found: {result_file}")
        
        # Load and analyze results
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total_segments = data['video_info']['total_segments']
        completed_segments = len(data['segments'])
        
        print(f"Progress: {completed_segments}/{total_segments} segments completed")
        print(f"Success rate: {completed_segments/total_segments*100:.1f}%")
        
        if completed_segments > 0:
            # Calculate average processing time
            first_segment = data['segments'][0]
            last_segment = data['segments'][-1]
            
            # Estimate time (rough calculation)
            total_video_time = last_segment['end_time'] - first_segment['start_time']
            
            print(f"Analyzed video time: {total_video_time:.1f} seconds")
            
            # Show sample results
            print(f"\nSample results:")
            for i, segment in enumerate(data['segments'][:3]):
                print(f"\nSegment {i+1} ({segment['start_time']:.1f}s-{segment['end_time']:.1f}s):")
                
                # Parse analysis JSON
                try:
                    analysis = json.loads(segment['analysis'])
                    print(f"  Type: {analysis.get('content_type', 'N/A')}")
                    print(f"  Keywords: {', '.join(analysis.get('keywords', [])[:5])}")
                    print(f"  Scene change: {analysis.get('scene_change', False)}")
                except:
                    print(f"  Raw analysis: {segment['analysis'][:100]}...")
        
    else:
        print("❌ Phase 2 result file not found")
        print("Check if Phase 2 is still running...")
    
    # Check Phase 1 data
    phase1_file = Path('output/news_phase1_data.pkl')
    if phase1_file.exists():
        print(f"\n✓ Phase 1 data available: {phase1_file}")
    else:
        print(f"\n❌ Phase 1 data not found: {phase1_file}")
    
    print(f"\nMonitoring completed at: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    monitor_progress()