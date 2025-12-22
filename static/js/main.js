// main.js - small helper; currently minimal but left for future UX enhancements
document.addEventListener('DOMContentLoaded', function(){
  // Basic progressive enhancement: focus first input on recommend page
  const firstInput = document.querySelector('input[type=number]');
  if(firstInput) firstInput.focus();
});
