from searchtweets import load_credentials, gen_rule_payload, collect_results, ResultStream
import requests

def connect_to_endpoint(url, headers, params, next_token = None):
    params['next_token'] = next_token   #params object received from create_url function
    response = requests.request("GET", url, headers = headers, params = params)
    print("Endpoint Response Code: " + str(response.status_code))
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()


enterprise_search_args = {'bearer_token': 'AAAAAAAAAAAAAAAAAAAAAL7hOgEAAAAAvM92PZSwVJ%2Ba%2BOD5Pgi4N298uTI%3DBBY7UCntIx9eXHBqGRjgjQcjoDFlMgFGJCjzd65uKISX8VFpwc',
                          'endpoint': 'https://api.twitter.com/2/tweets/search/all'}

#load_credentials(filename="twitter_keys.yaml", yaml_key="search_tweets_api", env_overwrite=False)


rule = gen_rule_payload("beyonce") # testing with a sandbox account

rs = ResultStream(rule_payload=rule,
                     max_results=10,
                     max_pages=1,
                     **enterprise_search_args) # change this if you need to

print(rs)
tweets = list(rs.stream())

#[print(tweet.all_text, end='\n\n') for tweet in tweets[0:10]]