def find_words_in_file(file_path, search_words, case_sensitive=False):
    """
    Find lines containing all specified words regardless of sequence.
    
    Args:
        file_path: Path to the file to search
        search_words: List of words to find
        case_sensitive: Boolean to control case sensitivity
    """
    # Prepare search words
    if not case_sensitive:
        for word in search_words:
            w1 = word.lower()
            if w1 == "200":
                w1 = " 200 "
            elif w1 == "400":
                w1 = " 400 "
            elif w1 == "404":
                w1 = " 404 "
            elif w1 == "500":
                w1 = " 500 "
            search_words.append(w1)
        #search_words = [word.lower() for word in search_words]
    
    matches = []
    
    try:
        with open(file_path, 'r') as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                # Convert line to lowercase if case-insensitive search
                check_line = line if case_sensitive else line.lower()
                
                # Check if all words are present in the line
                if all(word in check_line for word in search_words):
                    matches.append({
                        'line_number': line_num,
                        'content': line,
                        'matched_words': search_words
                    })
        
        return matches
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return []
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return []

def print_matches(matches, i, f):
    """Print matched lines with formatting"""
    if not matches:
        print("No matches found")
        return
    
    #print("\nMatches found:")
    #print("Line #  | Content")
    #print("-" * 80)
    
    for match in matches:
        #print(f"{match['line_number']:<7} | {match['content']}")
        write_line = str(i) + " " + str(match['line_number']) + " " + "1" + "\n"
        f.write(write_line)
    
    print(f"\nTotal matches found: {len(matches)}")

def main():
    # Get file path and search words from user
    file_path = "data/logs/aws-service-logs.txt"
    #search_input = input("Enter words to search (space-separated): ")
    #search_path = "logs/aws-service-queries.txt"
    #search_path = "logs/aws-service-logs-queries.txt"
    search_path = "data/logs/aws-service-logs-queries.txt"
    #search_input = "DELETE /home 500 25/Nov/2024"
    case_sensitive =  'n'
    i = 0

    with open('data/logs/aws-service-logs-qrels.txt', 'w') as f:

         for line in open(search_path, 'r'):
             search_input = line.strip()
             # Convert search input to list of words and remove empty strings
             search_words = [word.strip() for word in search_input.split() if word.strip()]
             
             if not search_words:
                 print("No valid search words provided")
                 return
             
             print(f"\nSearching for: {', '.join(search_words)}")
             matches = find_words_in_file(file_path, search_words, case_sensitive)
             print_matches(matches, i, f)
             i = i + 1
             #f.write(write_line)

if __name__ == "__main__":
    main()
