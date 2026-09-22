import { useEffect, useRef, useState } from 'react';
import { Link, NavLink, Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import './App.css';
import { AppIcon } from './components/app-icon';
import { HistoryPage } from './components/history-page';
import { ResetPasswordForm } from './components/reset-password-form';
import { ScreeningForm } from './components/screening-form';
import { SignInForm } from './components/sign-in-form';
import { SignUpForm } from './components/sign-up-form';
import { ThemeToggle } from './components/theme-toggle';
import { exchangeAuthCode, getAuthConfig } from './lib/auth';
import { useAuth } from './lib/auth-context';

type PublicConfig = { oAuthProviders: string[]; requireEmailVerification: boolean; passwordMinLength: number };

function Brand() {
  return <Link to="/" className="brand"><span className="brand-mark"><AppIcon name="eye" size={23}/></span><span><strong>RetinaSathi</strong><small>दृष्टि सुरक्षा साथी</small></span></Link>;
}

function AppLayout({ children }: { children: React.ReactNode }) {
  const { viewer, isLoading, signOut } = useAuth();
  const [online, setOnline] = useState(() => navigator.onLine);
  useEffect(() => {
    const update = () => setOnline(navigator.onLine);
    window.addEventListener('online', update);
    window.addEventListener('offline', update);
    return () => { window.removeEventListener('online', update); window.removeEventListener('offline', update); };
  }, []);
  return <main className="app-page"><header className="site-header"><div className="header-inner"><Brand/><nav className="site-nav" aria-label="Primary navigation"><NavLink to="/">Overview</NavLink>{viewer.isAuthenticated ? <><NavLink to="/screen">New screening</NavLink><NavLink to="/history">History</NavLink></> : null}</nav><div className="header-actions"><span className="connectivity"><i/>{online ? 'InsForge live' : 'Offline / local'}</span>{isLoading ? null : viewer.isAuthenticated ? <button className="button button--ghost" type="button" onClick={() => void signOut()}>Sign out</button> : <Link className="button button--primary button--small" to="/auth/sign-in">Sign in</Link>}</div></div></header>{!online ? <div className="offline-banner"><AppIcon name="wifi" size={16}/>Internet unavailable. A running local model can still analyze; secure cloud saves will queue in this browser.</div> : null}{children}<footer className="site-footer"><div><Brand/><p>Explainable diabetic-retinopathy screening support for frontline health workers.</p></div><div><span>Powered by InsForge</span><span>Research prototype · SIH26038</span><ThemeToggle/></div></footer></main>;
}

function HomePage() {
  const localModel = import.meta.env.VITE_INFERENCE_MODE?.trim() === 'local';
  const v34 = import.meta.env.VITE_MODEL_GENERATION?.trim() === 'v3.4';
  return <AppLayout><div className="home-page"><section className="hero"><div className="hero-copy"><p className="eyebrow">Explainable AI · Built for rural India</p><h1>Catch diabetic eye disease <em>before sight is lost.</em></h1><p className="hero-lead">RetinaSathi helps frontline health workers assess fundus photographs, understand the AI’s evidence, and refer high-risk patients sooner.</p><div className="hero-actions"><Link className="button button--primary" to="/screen">Start a screening <AppIcon name="arrow" size={18}/></Link><a className="button button--secondary" href="#how-it-works">See how it works</a></div><div className="trust-row"><span><AppIcon name="shield" size={17}/>Private patient records</span><span><AppIcon name="wifi" size={17}/>Low-bandwidth workflow</span></div></div><div className="hero-visual" aria-label="RetinaSathi workflow illustration"><div className="retina-orbit"><div className="retina-disc"><span/><i/><b/></div><div className="scan-line"/></div><div className="floating-card floating-card--grade"><small>DR severity</small><strong>Grade 2</strong><span>Moderate · Refer</span></div><div className="floating-card floating-card--quality"><span className="status-dot"/><div><small>Image quality</small><strong>Suitable for screening</strong></div></div></div></section>
  <section className="impact-strip"><div><strong>0–4</strong><span>DR severity grading</span></div><div><strong>0–2</strong><span>Macular-edema risk</span></div><div><strong>Explicit</strong><span>Module readiness</span></div><div><strong>RLS</strong><span>User-isolated records</span></div></section>
  <section className="how-section" id="how-it-works"><div className="section-intro"><p className="eyebrow">Designed around clinical action</p><h2>Not just a score. A reason to act.</h2><p>Each result combines image quality, disease severity, uncertainty, and visual evidence in one workflow.</p></div><div className="feature-grid"><article><span className="feature-index">01</span><AppIcon name="scan" size={27}/><h3>Quality before grading</h3><p>Ungradeable captures stop inference and receive simple retake instructions.</p></article><article><span className="feature-index">02</span><AppIcon name="eye" size={27}/><h3>Two screening signals</h3><p>The deployed V1 estimates DR grade and DME risk with explicit limitations.</p></article><article><span className="feature-index">03</span><AppIcon name="alert" size={27}/><h3>Visible evidence</h3><p>Attention overlays show model influence and are never labelled as confirmed lesions.</p></article><article><span className="feature-index">04</span><AppIcon name="shield" size={27}/><h3>Privacy by default</h3><p>InsForge Auth, row-level security, and private storage isolate operator records.</p></article></div></section>
  <section className="model-card-section"><div><p className="eyebrow">Transparent by design</p><h2>An honest model card is part of the product.</h2><p>{v34 ? 'Local V3.4 is a calibrated research screening candidate. It passed three DeepDRiD source-validation repeats, but exact five-grade performance remains limited and every output requires clinician review.' : localModel ? 'The local model is a research candidate with calibrated DR probabilities and visual influence evidence. Every result still requires clinician review.' : 'The deployed V1 is a prototype baseline, not a clinical model. Local testing uses stronger candidates while the cloud service is being replaced.'}</p></div><dl><div><dt>Dataset</dt><dd>{v34 ? 'APTOS + IDRiD + DeepDRiD' : localModel ? 'APTOS + IDRiD' : 'IDRiD, India'}</dd></div><div><dt>Architecture</dt><dd>{v34 ? 'Partial DINOv2-S/14' : localModel ? 'EfficientNet-B3' : 'MobileNetV3 Small'}</dd></div><div><dt>Outputs</dt><dd>{v34 ? 'Referable DR + five grades + patch influence' : localModel ? 'DR + calibration + visual evidence' : 'DR + DME + attention'}</dd></div><div><dt>Deployment</dt><dd>{localModel ? 'Local MPS / CPU' : 'Cloud prototype'}</dd></div></dl></section></div></AppLayout>;
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { viewer, isLoading } = useAuth();
  if (isLoading) return <div className="auth-callback-loading"><span className="spinner spinner--dark"/></div>;
  return viewer.isAuthenticated ? children : <Navigate to="/auth/sign-in" replace/>;
}

function ScreeningPage() { return <AppLayout><div className="workspace-page"><ScreeningForm/></div></AppLayout>; }
function ScreeningHistoryPage() { return <AppLayout><div className="workspace-page"><HistoryPage/></div></AppLayout>; }

function AuthPageShell({ children, footer }: { children: React.ReactNode; footer: React.ReactNode }) {
  return <main className="auth-page"><div className="auth-art"><Brand/><div><p className="eyebrow">A clearer path to referral</p><h2>Every screening deserves an explanation.</h2><p>Securely screen, review, and follow up with patients from one shared clinical workspace.</p></div><small>Research prototype · Not a medical diagnosis</small></div><div className="auth-shell"><div className="auth-mobile-brand"><Brand/></div><div className="auth-card">{children}<div className="auth-footer">{footer}</div></div></div></main>;
}
function SignInPage({ providers }: { providers: string[] }) { return <AuthPageShell footer={<p>New to RetinaSathi? <Link to="/auth/sign-up">Create an account</Link></p>}><div className="auth-header"><p className="eyebrow">Secure workspace</p><h1>Welcome back</h1><p>Sign in to continue screening.</p></div><SignInForm providers={providers}/></AuthPageShell>; }
function SignUpPage({ providers }: { providers: string[] }) { return <AuthPageShell footer={<p>Already registered? <Link to="/auth/sign-in">Sign in</Link></p>}><SignUpForm providers={providers}/></AuthPageShell>; }
function ResetPasswordPage() { return <AuthPageShell footer={<p><Link to="/auth/sign-in">Back to sign in</Link></p>}><ResetPasswordForm/></AuthPageShell>; }

function AuthCallbackPage() {
  const navigate = useNavigate();
  const { viewer, isLoading, refreshViewer } = useAuth();
  const [error, setError] = useState('');
  const exchangeStarted = useRef(false);
  useEffect(() => { if (viewer.isAuthenticated) window.location.replace('/screen'); }, [viewer.isAuthenticated]);
  useEffect(() => {
    if (isLoading || exchangeStarted.current) return;
    const code = new URL(window.location.href).searchParams.get('insforge_code');
    if (!code) { if (!viewer.isAuthenticated) navigate('/auth/sign-in', { replace: true }); return; }
    exchangeStarted.current = true;
    void (async () => { const result = await exchangeAuthCode(code); if (result.success) { await refreshViewer(); window.location.replace('/screen'); } else setError(result.error); })();
  }, [isLoading, navigate, refreshViewer, viewer.isAuthenticated]);
  if (error) return <AuthPageShell footer={<p><Link to="/auth/sign-in">Back to sign in</Link></p>}><div className="auth-header"><h1>Unable to sign in</h1><p>{error}</p></div></AuthPageShell>;
  return <div className="auth-callback-loading"><span className="spinner spinner--dark"/></div>;
}

export default function App() {
  const [config, setConfig] = useState<PublicConfig>({ oAuthProviders: [], requireEmailVerification: false, passwordMinLength: 8 });
  useEffect(() => { void getAuthConfig().then((next) => setConfig({ oAuthProviders: next.oAuthProviders ?? [], requireEmailVerification: next.requireEmailVerification ?? false, passwordMinLength: next.passwordMinLength ?? 8 })); }, []);
  return <Routes><Route path="/" element={<HomePage/>}/><Route path="/screen" element={<ProtectedRoute><ScreeningPage/></ProtectedRoute>}/><Route path="/history" element={<ProtectedRoute><ScreeningHistoryPage/></ProtectedRoute>}/><Route path="/auth/sign-in" element={<SignInPage providers={config.oAuthProviders}/>}/><Route path="/auth/sign-up" element={<SignUpPage providers={config.oAuthProviders}/>}/><Route path="/auth/reset-password" element={<ResetPasswordPage/>}/><Route path="/auth/callback" element={<AuthCallbackPage/>}/><Route path="*" element={<Navigate to="/" replace/>}/></Routes>;
}
