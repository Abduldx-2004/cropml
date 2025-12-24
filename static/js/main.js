// main.js - Enhanced UX features for Crop Recommendation System

document.addEventListener('DOMContentLoaded', function(){
  // Focus first input on recommend page
  const firstInput = document.querySelector('input[type=number]');
  if(firstInput) firstInput.focus();

  // Add loading animation to form submission
  const form = document.getElementById('recommendForm');
  if(form) {
    form.addEventListener('submit', function(e) {
      const submitBtn = form.querySelector('button[type="submit"]');
      if(submitBtn) {
        submitBtn.innerHTML = '🔄 Analyzing...';
        submitBtn.disabled = true;
        submitBtn.style.opacity = '0.7';
      }
    });
  }

  // Add smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if(target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Add fade-in animation for cards
  const cards = document.querySelectorAll('.card');
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if(entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, observerOptions);

  cards.forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(card);
  });

  // Add input validation feedback
  const inputs = document.querySelectorAll('input[type="number"]');
  inputs.forEach(input => {
    input.addEventListener('input', function() {
      const value = parseFloat(this.value);
      const min = parseFloat(this.min) || 0;
      const max = parseFloat(this.max) || Infinity;

      if(this.value && (value < min || value > max)) {
        this.style.borderColor = '#f44336';
        this.style.boxShadow = '0 0 0 3px rgba(244, 67, 54, 0.1)';
      } else {
        this.style.borderColor = '';
        this.style.boxShadow = '';
      }
    });

    // Add placeholder animation
    input.addEventListener('focus', function() {
      this.parentElement.style.transform = 'scale(1.02)';
    });

    input.addEventListener('blur', function() {
      this.parentElement.style.transform = 'scale(1)';
    });
  });

  // Add confidence meter animation on result page
  const confidenceElement = document.querySelector('.confidence');
  if(confidenceElement) {
    const confidenceText = confidenceElement.textContent;
    const confidenceMatch = confidenceText.match(/(\d+\.?\d*)%/);
    if(confidenceMatch) {
      const confidenceValue = parseFloat(confidenceMatch[1]);
      confidenceElement.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
          <div style="flex: 1; height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden;">
            <div style="height: 100%; background: linear-gradient(90deg, #2e7d32, #66bb6a); width: 0%; border-radius: 4px; transition: width 2s ease;"></div>
          </div>
          <span>${confidenceValue}%</span>
        </div>
      `;

      setTimeout(() => {
        const progressBar = confidenceElement.querySelector('div > div > div');
        if(progressBar) {
          progressBar.style.width = `${confidenceValue}%`;
        }
      }, 500);
    }
  }
});
