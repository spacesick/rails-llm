module PromptService
  extend ActiveSupport::Concern

  def initialize
    @conn = Faraday.new(url: 'http://127.0.0.1:3000/')
  end

  def post_prompt(prompt)
    response = @conn.post('/generate') do |req|
      req.headers['Content-Type'] = 'application/json'
      req.body = { prompt: prompt }.to_json
    end
    JSON.parse(response.body)[0]
  end
end
