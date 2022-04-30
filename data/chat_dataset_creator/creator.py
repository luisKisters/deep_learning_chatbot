file_lines = []
new_intents = []


# THX DUDE ON STACK FOR THIS <3 https://stackoverflow.com/questions/9573244/how-to-check-if-the-string-is-empty
def is_not_blank(s):
    return bool(s and not s.isspace())

try:
    f = open("./structure.txt", "r")
    lines = f.readline()
    while lines:
        file_lines.append(lines)
        lines = f.readline()
finally:
    f.close()

try:
    pattern_num = int(input("Enter how many patterns you want to create (max: 10): "))
except ValueError:
    print("Wrong input! Please enter a number")
    pattern_num = int(input("Enter how many patterns you want to create (max: 10): "))

try:
    responses_num = int(input("Enter how many responses you want to create: "))
except ValueError:      
    print("Wrong input! Please enter a number")
    responses_num = int(input("Enter how many responses you want to create: ")) 

for line in file_lines:  
    if "$TAG" in line:
        input_tag = input("Enter the tag: ")
        if input_tag == "":
            print("Tag is empty, please enter a tag")
            input_tag = input("Enter the tag: ")
            if input_tag == "":
                print("AGAIN A EMPTY TAG?!?!?! OK THAN U WILL GET A FCKED UP JSON, IDC")
            new_tag = line.replace("$TAG", input_tag)           
            new_intents.append(new_tag)    
        else:
            new_tag = line.replace("$TAG", input_tag)           
            new_intents.append(new_tag)
    
    elif "$PATTERN" in line:
        if pattern_num > 0:
            input_pattern = input("Enter the pattern: ")
            if input_pattern == "":
                pattern_num -= 1
            else:
                new_pattern = line.replace("$PATTERN", input_pattern)           
                new_intents.append(new_pattern)
                pattern_num -= 1
        else: 
            pass

    elif "$RESPONSE" in line:
        if responses_num > 0:
            input_response = input("Enter the response: ")
            if input_response == "":
                responses_num -= 1
            else:
                new_response = line.replace("$RESPONSE", input_response)           
                new_intents.append(new_response)
                responses_num -= 1
        else:
            pass
        
    else:
        new_intents.append(line)

with open ("output.txt", "a") as file:
    file.write("\n")
    for line in new_intents:
        file.write(line)
