import { Controller } from "@hotwired/stimulus"

// Connects to data-controller="scroll"
export default class extends Controller {
  static targets = ['responses']

  connect() {
    this.scrollToBottom();
    this.registerMutationObserver();
  }

  scrollToBottom() {
    if (this.responsesTarget.scrollHeight > this.responsesTarget.clientHeight) {
      this.responsesTarget.scrollTop = this.responsesTarget.scrollHeight;
    }
  }

  registerMutationObserver() {
    const observer = new MutationObserver(() => {
      this.scrollToBottom();
    });

    observer.observe(this.responsesTarget, { childList: true });
  }
}
