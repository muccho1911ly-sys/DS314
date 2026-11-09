import { useEffect, useRef, useState } from "react";
import { Check, ChevronUp, Cpu } from "lucide-react";

export default function FloatingModelPicker({ models, model, onModelChange, queueCount, onOpenQueue }) {
  const [open, setOpen] = useState(false);
  const host = useRef(null);
  const trigger = useRef(null);
  useEffect(() => {
    if (!open) return;
    const outside = event => { if (!host.current?.contains(event.target)) setOpen(false); };
    const escape = event => { if (event.key === "Escape") { setOpen(false); trigger.current?.focus(); } };
    document.addEventListener("pointerdown", outside);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  return <div className="floating-model-picker" ref={host}>
    {open && <section id="floating-model-options" className="floating-model-options" aria-label="Choose search model">
      <strong>Search model</strong>
      <div role="group" aria-label="Available models">
        {models.map(key => <button key={key} type="button" className="floating-model-option" aria-pressed={model === key}
          onClick={() => { onModelChange(key); setOpen(false); trigger.current?.focus(); }}>
          <span>{key}</span>{model === key && <Check size={16} />}
        </button>)}
        {!models.length && <p>No models available. Check backend connection.</p>}
      </div>
      <button type="button" className="floating-model-queue" onClick={() => { setOpen(false); onOpenQueue(); }}>Open Queue ({queueCount})</button>
    </section>}
    <button ref={trigger} type="button" className="floating-model-trigger" aria-expanded={open} aria-controls="floating-model-options"
      title={model ? `Current model: ${model}` : "Choose search model"} onClick={() => setOpen(value => !value)}>
      <Cpu size={18} /><span>Model</span><ChevronUp size={16} />
    </button>
  </div>;
}
