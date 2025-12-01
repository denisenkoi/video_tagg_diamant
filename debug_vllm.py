"""
Debug VLLM - отладка конкретных проблемных сегментов
"""

from phase2_vllm_analysis import Phase2VLLMAnalyzer
import json

def debug_segment(segment_index: int):
    """Debug specific segment"""
    print(f"=== Debugging segment {segment_index} ===")
    
    # Initialize analyzer
    analyzer = Phase2VLLMAnalyzer()
    
    # Load Phase 1 data
    segment_data = analyzer.load_phase1_data('news')
    
    if segment_index >= len(segment_data):
        print(f"❌ Segment {segment_index} not found (max: {len(segment_data)-1})")
        return
    
    # Get specific segment
    data = segment_data[segment_index]
    segment = data['segment']
    transcript = data['transcript']
    
    print(f"Segment info:")
    print(f"  Time: {segment['start']:.1f}s - {segment['end']:.1f}s")
    print(f"  Transcript: {transcript[:100]}...")
    
    # Get frames
    base64_frames = analyzer.get_base64_frames(data)
    print(f"  Frames: {len(base64_frames)}")
    
    # Create prompt
    previous_context = ""  # Empty for debugging
    prompt = analyzer.create_vllm_prompt(previous_context, transcript)
    
    # Test VLLM request
    print(f"\nTesting VLLM request...")
    analysis = analyzer.analyze_segment_vllm(base64_frames, prompt)
    
    if analysis:
        print(f"✓ VLLM returned response")
        print(f"Raw response length: {len(analysis)}")
        print(f"\nRaw response:")
        print("="*50)
        print(analysis)
        print("="*50)
        
        # Test JSON extraction
        print(f"\nTesting JSON extraction...")
        clean_json = analyzer.extract_json_from_response(analysis)
        print(f"Clean JSON length: {len(clean_json)}")
        print(f"\nClean JSON:")
        print("-"*30)
        print(clean_json)
        print("-"*30)
        
        # Test JSON parsing
        try:
            parsed = json.loads(clean_json)
            print(f"\n✓ JSON parsing successful!")
            print(f"Keys: {list(parsed.keys())}")
            
            # Check required fields
            required_fields = ['description', 'keywords', 'confidence']
            missing = [f for f in required_fields if f not in parsed]
            if missing:
                print(f"⚠️ Missing fields: {missing}")
            else:
                print("✓ All required fields present")
                
        except Exception as e:
            print(f"\n❌ JSON parsing failed: {e}")
            print(f"Clean JSON repr: {repr(clean_json)}")
    else:
        print(f"❌ VLLM returned no response")

if __name__ == "__main__":
    # Debug the problematic segment 3
    debug_segment(2)  # 0-indexed, so segment 3 is index 2