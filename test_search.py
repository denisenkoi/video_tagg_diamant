"""
Test ChromaDB search functionality
Simple script to verify that data was loaded and search works
"""
import sys
from db_manager import ChromaDBManager


def format_time(seconds: float) -> str:
    """Format seconds to MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def print_result(result: dict, index: int):
    """Pretty print a search result"""
    print(f"\n{'='*80}")
    print(f"Result #{index + 1}")
    print(f"{'='*80}")
    print(f"Video: {result['video_name']}")
    print(f"Segment: {result['segment_index']}")
    print(f"Time: {format_time(result['start_time'])} - {format_time(result['end_time'])} ({result['duration']:.1f}s)")
    print(f"Relevance: {result['relevance_score']:.2%}")
    print(f"Confidence: {result['confidence']}")
    print(f"\nDescription:")
    print(f"  {result['description']}")
    print(f"\nKeywords:")
    print(f"  {result['keywords']}")
    print(f"\nVideo path: {result['video_path']}")


def main():
    """Main test function"""
    print("=== ChromaDB Search Test ===\n")

    # Initialize ChromaDB
    print("Initializing ChromaDB...")
    db_manager = ChromaDBManager(
        persist_directory="./chroma_db",
        collection_name="video_segments"
    )

    # Show stats
    stats = db_manager.get_collection_stats()
    print(f"\n📊 Collection Stats:")
    print(f"  Collection: {stats['collection_name']}")
    print(f"  Total segments: {stats['total_segments']}")
    print(f"  Persist directory: {stats['persist_directory']}")

    if stats['total_segments'] == 0:
        print("\n❌ No segments in database!")
        print("Run load_data_to_chroma.py first to load Phase 2 data.")
        return 1

    # Test queries
    test_queries = [
        "новости",
        "авария на дороге",
        "люди разговаривают",
        "автомобиль",
        "news anchor",
        "weather forecast"
    ]

    print(f"\n{'='*80}")
    print("Running test queries...")
    print(f"{'='*80}")

    for query in test_queries:
        print(f"\n\n🔍 Query: '{query}'")
        print("-" * 80)

        results = db_manager.search(
            query=query,
            limit=3
        )

        if not results:
            print("  No results found")
            continue

        print(f"  Found {len(results)} results")

        for i, result in enumerate(results):
            print_result(result, i)

    # Interactive search
    print(f"\n\n{'='*80}")
    print("Interactive Search Mode")
    print(f"{'='*80}")
    print("Enter search queries (or 'quit' to exit):\n")

    while True:
        try:
            query = input("Search> ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not query:
                continue

            print(f"\n🔍 Searching for: '{query}'")
            print("-" * 80)

            results = db_manager.search(query=query, limit=5)

            if not results:
                print("No results found\n")
                continue

            print(f"Found {len(results)} results:\n")

            for i, result in enumerate(results):
                print(f"{i+1}. [{result['video_name']}] "
                      f"{format_time(result['start_time'])}-{format_time(result['end_time'])} "
                      f"(rel: {result['relevance_score']:.2%})")
                print(f"   {result['description'][:150]}...")
                print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
