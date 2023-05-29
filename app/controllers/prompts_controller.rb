class PromptsController < ApplicationController

include PromptService

  def create
    @text = params[:prompt]
    @response = post_prompt(params[:prompt])

    @prompt = Prompt.create(text: @text, response: @response)

    respond_to do |format|
      format.turbo_stream
      format.html {
        redirect_to root_path,
        flash: { scroll_to_bottom: true }
      }
    end
  end

end
