from rag import retrieve_chunks, estimate_line_numbers, get_chunk_with_context

# 2. Search
quit = False
while not quit:
    query = input("Enter your query: ")
    if query.lower() == "quit":
        quit = True
        break
    try:
        top_k = int(input("Enter the number of results to display: "))
    except ValueError:
        print("Please enter a valid integer for the number of results.")
        continue

    hits = retrieve_chunks(query, top_k=top_k)

    print("=" * 80)
    print(f"📄 SEARCH RESULTS: Found {len(hits)} chunks")
    print("=" * 80)

    for i, h in enumerate(hits, 1):
        print(f"\n🔍 RESULT #{i}")
        print("─" * 60)
        print(f"📊 Score: {h['score']:.4f}")
        print(f"📁 Source: {h['source']}")
        
        # Calculate and show line numbers
        if h['start_char_idx'] is not None and h['end_char_idx'] is not None:
            estimated_lines = estimate_line_numbers(h['source'], h['start_char_idx'], h['end_char_idx'])
            print(f"📏 Lines: {estimated_lines}")
            
            # Show content with proper context and line numbers
            print("─" * 60)
            print("📝 Content:")
            context_content = get_chunk_with_context(h['source'], h['start_char_idx'], h['end_char_idx'], context_lines=2)
            
            # Add line numbers to each line
            lines = context_content.split('\n')
            try:
                start_line_num = int(estimated_lines.split('-')[0])
            except Exception:
                start_line_num = 1
            
            for line_idx, line in enumerate(lines):
                line_num = start_line_num + line_idx
                if line.startswith('>>> '):
                    # This is the actual chunk content
                    print(f"{line_num:4d}: {line[4:]}")  # Remove the >>> prefix
                elif line.startswith('    '):
                    # This is context
                    print(f"{line_num:4d}: {line[4:]}")  # Remove the context prefix
                else:
                    print(f"{line_num:4d}: {line}")
        else:
            print(f"📏 Lines: {h['line_range']}")
            print("─" * 60)
            print("📝 Content:")
            # Just show the raw content if we don't have character indices
            lines = h['chunk'].split('\n')
            for line_idx, line in enumerate(lines, 1):
                print(f"{line_idx:4d}: {line}")
        
        print("=" * 80)

    print(f"\n✅ Search completed - {len(hits)} results displayed")
