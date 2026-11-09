export default function HeroDecoration() {
  return (
    <div className="hero-decoration" aria-hidden="true">
      <div className="hero-orbits">
        <i /><i /><i /><i />
      </div>
      <svg className="hero-flow hero-flow-left" viewBox="0 0 240 340" fill="none">
        <path d="M-40 0C210 60-80 180 120 340" />
        <path d="M-75 0C175 60-115 180 85 340" />
      </svg>
      <svg className="hero-flow hero-flow-right" viewBox="0 0 240 340" fill="none">
        <path d="M260 0C40 80 300 180 100 340" />
      </svg>
      <div className="hero-dots" />
    </div>
  );
}
