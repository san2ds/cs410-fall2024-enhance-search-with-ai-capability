import time
from datetime import datetime, timedelta

def create_access_log():
    with open('logs/access.log.1', 'w') as f:
        # Sample IP addresses
        ips = ['192.168.1.100', '10.0.0.1', '172.16.0.20']
        # Sample URLs
        urls = ['/home', '/api/users', '/login', '/logout', '/products']
        # Sample user agents
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
        ]
        
        # Generate sample logs
        for i in range(35000):
            ip = ips[i % len(ips)]
            timestamp = (((datetime.now() + timedelta(days=i / 2500)) + timedelta(seconds=1 if i % 2 != 0 else 2))).strftime('%d/%b/%Y:%H:%M:%S +0000') 
            method = 'GET' if i % 3 != 0 else 'PUT' if i % 4 != 0 else 'POST' if i % 8 != 0 else 'DELETE'
            url = urls[i % len(urls)]
            status = '200' if i % 4 != 0 else '404' if i % 3 != 0 else '400' if i % 7 != 0 else '500'
            success_msg = "Operation successful " + url
            error = ""
            if status == '404':
                error_msg = "- ResourceNotFoundException: Resource not found" if i % 5 !=0 else "- UnauthorizedException: Unauthorized" 
            elif status == '400':
                error_msg = "- InvalidParameterException: Missing required field" if i % 8 !=0 else "- OutOfMemoryError: Java heap space"
            elif status == '500':
                error_msg = "- InternalServerError: Internal server error"
            error = method + " " + url + error_msg
            
            msg = success_msg if status == '200' else error
            bytes_sent = str(2000 + i)
            user_agent = user_agents[i % len(user_agents)]
            
            log_line = f'{ip} - - [{timestamp}] "{method} {url} HTTP/1.1" {status} {bytes_sent} "-" "{user_agent}" "{msg}"\n'
            f.write(log_line)
            #time.sleep(0.1)  # Add small delay between entries

create_access_log()
