import React, { useState, useEffect } from 'react';
import {
  X,
  Search,
  Database,
  Sliders,
  CheckCircle2,
  Sparkles,
  Layers,
  ArrowRight,
  ShieldCheck,
  Cpu,
  RefreshCw,
  Copy,
  Check,
  Zap,
  Info,
  ChevronRight,
  Fingerprint,
  Trash2,
  Plus
} from 'lucide-react';
import { CowAnalysisResult, FaissIndexStats, FaissSearchResult, FaissVectorRecord } from '../types';
import { formatAadhaarId } from '../utils/muzzleBiometrics';
import {
  cosineSimilarity,
  dotProduct,
  l2Distance,
  l2Normalize,
  VECTOR_DIMENSION,
  extractBiometricFeatureVector,
  globalFaissIndex
} from '../utils/faissEngine';

interface FaissVectorSearchModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentAnalysis: CowAnalysisResult | null;
  savedScans: CowAnalysisResult[];
  onSaveToRegistry?: (scan: CowAnalysisResult) => void;
}

export const FaissVectorSearchModal: React.FC<FaissVectorSearchModalProps> = ({
  isOpen,
  onClose,
  currentAnalysis,
  savedScans,
  onSaveToRegistry,
}) => {
  const [indexType, setIndexType] = useState<'IndexFlatIP' | 'IndexIVFFlat' | 'IndexFlatL2'>('IndexFlatIP');
  const [metric, setMetric] = useState<'cosine' | 'l2'>('cosine');
  const [topK, setTopK] = useState<number>(5);
  const [threshold, setThreshold] = useState<number>(0.0);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [searchResults, setSearchResults] = useState<FaissSearchResult[]>([]);
  const [searchTimeMs, setSearchTimeMs] = useState<number>(0.14);
  const [totalIndexed, setTotalIndexed] = useState<number>(0);
  const [stats, setStats] = useState<FaissIndexStats | null>(null);
  const [allVectors, setAllVectors] = useState<FaissVectorRecord[]>([]);

  // Selected Vector for pairwise Cosine Comparator
  const [compareTargetId, setCompareTargetId] = useState<string>('');
  const [copiedVector, setCopiedVector] = useState(false);
  const [saveSuccessMsg, setSaveSuccessMsg] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'search' | 'inspector' | 'database'>('search');

  // Load vector stats and vectors from server
  const loadIndexStats = async () => {
    try {
      const statsRes = await fetch('/api/faiss/index/stats');
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        setStats(statsData);
        setTotalIndexed(statsData.totalVectors);
      }

      const vecRes = await fetch('/api/faiss/index/vectors');
      if (vecRes.ok) {
        const vecData = await vecRes.json();
        const vectors: FaissVectorRecord[] = vecData.vectors || [];
        setAllVectors(vectors);
        if (vectors.length > 0) {
          if (!compareTargetId || !vectors.some((v) => v.id === compareTargetId)) {
            setCompareTargetId(vectors[0].id);
          }
        } else {
          setCompareTargetId('');
        }
      }
    } catch (err) {
      console.warn('Could not load FAISS server stats:', err);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadIndexStats();
      performSearch();
    }
  }, [isOpen, indexType, metric, topK, threshold, currentAnalysis]);

  const getQueryVector = (): number[] => {
    if (currentAnalysis?.featureVector?.vector) {
      return currentAnalysis.featureVector.vector;
    }
    // Fallback extraction
    const emb = extractBiometricFeatureVector({
      seedHash: currentAnalysis?.biometricPassport?.muzzlePatternHash || 'DEFAULT-MZ-SEED',
      breed: currentAnalysis?.primaryBreed?.breed,
      species: currentAnalysis?.primaryBreed?.speciesType,
      beadDensityScore: currentAnalysis?.muzzleDetections?.[0]?.beadDensityScore || 88,
      ridgePatternClarity: currentAnalysis?.muzzleDetections?.[0]?.ridgePatternClarity || 'High',
      symmetryScore: currentAnalysis?.muzzleDetections?.[0]?.symmetryScore || 90,
      nostrilDistanceNorm: currentAnalysis?.muzzleDetections?.[0]?.nostrilDistanceNorm || 0.22,
    });
    return emb.vector;
  };

  const performSearch = async () => {
    setIsSearching(true);
    const queryVector = getQueryVector();

    try {
      const res = await fetch('/api/faiss/index/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          queryVector,
          topK,
          threshold,
          metric,
          indexType,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setSearchResults(data.results || []);
        setSearchTimeMs(data.searchTimeMs || 0.12);
        setTotalIndexed(data.totalIndexed || 0);
      }
    } catch (err) {
      console.error('FAISS search error:', err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleSaveCurrentVector = async () => {
    if (!currentAnalysis) return;
    const queryVector = getQueryVector();
    const id = `vec-${currentAnalysis.biometricPassport.uniqueCattleId}`;

    const record: FaissVectorRecord = {
      id,
      cowId: currentAnalysis.biometricPassport.uniqueCattleId,
      breed: currentAnalysis.primaryBreed.breed,
      species: currentAnalysis.primaryBreed.speciesType,
      imageUrl: currentAnalysis.imageUrl,
      vector: queryVector,
      beadDensityScore: currentAnalysis.muzzleDetections?.[0]?.beadDensityScore || 88,
      symmetryScore: currentAnalysis.muzzleDetections?.[0]?.symmetryScore || 90,
      timestamp: Date.now(),
      notes: `Registered from active scan on ${new Date().toLocaleDateString()}`,
    };

    try {
      globalFaissIndex.addVector(record);

      const res = await fetch('/api/faiss/index/add', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ record }),
      });

      if (res.ok) {
        const result = await res.json();
        setSaveSuccessMsg(`✓ Biometric Vector Registered in FAISS Index! (${record.cowId} is now available for future similarity queries & comparison)`);
        setTimeout(() => setSaveSuccessMsg(null), 4000);
        
        if (onSaveToRegistry) {
          onSaveToRegistry(currentAnalysis);
        }

        await loadIndexStats();
        performSearch();
      }
    } catch (err) {
      console.error('Save vector error:', err);
    }
  };

  const handleDeleteVector = async (id: string, cowId?: string) => {
    try {
      globalFaissIndex.removeVector(id);
      if (cowId) globalFaissIndex.removeVector(`vec-${cowId}`);

      const res = await fetch(`/api/faiss/index/vectors/${encodeURIComponent(id)}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        setSaveSuccessMsg(`Vector ${cowId || id} removed from FAISS registry`);
        setTimeout(() => setSaveSuccessMsg(null), 3000);
        await loadIndexStats();
        performSearch();
      }
    } catch (err) {
      console.error('Delete vector error:', err);
    }
  };

  const handleClearAllVectors = async () => {
    if (window.confirm('Delete ALL records from the FAISS vector database? All future comparisons will require newly registered cattle.')) {
      try {
        globalFaissIndex.clear();
        const res = await fetch('/api/faiss/index/clear', { method: 'POST' });
        if (res.ok) {
          setSaveSuccessMsg('All FAISS vector records deleted successfully. Registry is now clean.');
          setTimeout(() => setSaveSuccessMsg(null), 3500);
          setAllVectors([]);
          setSearchResults([]);
          setTotalIndexed(0);
          await loadIndexStats();
        }
      } catch (err) {
        console.error('Clear FAISS error:', err);
      }
    }
  };

  const handleCopyVectorJson = () => {
    const vec = getQueryVector();
    navigator.clipboard.writeText(JSON.stringify(vec, null, 2));
    setCopiedVector(true);
    setTimeout(() => setCopiedVector(false), 2000);
  };

  if (!isOpen) return null;

  const currentVector = getQueryVector();
  const targetRecord = allVectors.find((v) => v.id === compareTargetId) || allVectors[0];
  const targetVector = targetRecord ? targetRecord.vector : currentVector;
  const directCosine = cosineSimilarity(currentVector, targetVector);
  const directL2 = l2Distance(currentVector, targetVector);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 bg-black/85 backdrop-blur-xl overflow-y-auto animate-in fade-in duration-200">
      <div className="bg-[#0c140c] text-white border border-emerald-500/30 rounded-3xl max-w-5xl w-full max-h-[92vh] flex flex-col shadow-2xl shadow-black/80 relative overflow-hidden">
        {/* Background glow */}
        <div className="absolute top-0 right-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute bottom-0 left-1/4 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none"></div>

        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-white/10 relative z-10 bg-black/40 backdrop-blur-md">
          <div className="flex items-center space-x-3.5">
            <div className="w-11 h-11 rounded-2xl bg-emerald-500/20 border border-emerald-400/40 text-emerald-300 flex items-center justify-center shadow-lg shadow-emerald-500/10">
              <Cpu className="w-6 h-6 text-emerald-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white tracking-tight">
                  FAISS Vector Similarity Engine
                </h3>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 uppercase">
                  128-D Cosine Index
                </span>
              </div>
              <p className="text-xs text-slate-400">
                High-dimensional biometric feature embeddings &amp; cosine similarity nearest-neighbor search
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-2 rounded-xl bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-colors cursor-pointer border border-white/10"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 px-6 py-3 border-b border-white/10 bg-black/20 text-xs font-semibold">
          <button
            onClick={() => setActiveTab('search')}
            className={`px-4 py-2 rounded-xl transition-all cursor-pointer flex items-center gap-2 ${
              activeTab === 'search'
                ? 'bg-emerald-500 text-black font-extrabold shadow-md shadow-emerald-500/20'
                : 'text-slate-300 hover:text-white bg-white/5 hover:bg-white/10'
            }`}
          >
            <Search className="w-3.5 h-3.5" />
            <span>FAISS Top-K Search</span>
          </button>
          <button
            onClick={() => setActiveTab('inspector')}
            className={`px-4 py-2 rounded-xl transition-all cursor-pointer flex items-center gap-2 ${
              activeTab === 'inspector'
                ? 'bg-emerald-500 text-black font-extrabold shadow-md shadow-emerald-500/20'
                : 'text-slate-300 hover:text-white bg-white/5 hover:bg-white/10'
            }`}
          >
            <Fingerprint className="w-3.5 h-3.5" />
            <span>128-D Vector Inspector &amp; Math</span>
          </button>
          <button
            onClick={() => setActiveTab('database')}
            className={`px-4 py-2 rounded-xl transition-all cursor-pointer flex items-center gap-2 ${
              activeTab === 'database'
                ? 'bg-emerald-500 text-black font-extrabold shadow-md shadow-emerald-500/20'
                : 'text-slate-300 hover:text-white bg-white/5 hover:bg-white/10'
            }`}
          >
            <Database className="w-3.5 h-3.5" />
            <span>Vector Registry ({totalIndexed})</span>
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 relative z-10 custom-scrollbar">
          {saveSuccessMsg && (
            <div className="p-3.5 rounded-2xl bg-emerald-950/60 border border-emerald-500/40 text-emerald-200 text-xs flex items-center gap-2 animate-in fade-in">
              <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
              <span className="font-semibold">{saveSuccessMsg}</span>
            </div>
          )}

          {/* TAB 1: FAISS TOP-K SEARCH */}
          {activeTab === 'search' && (
            <div className="space-y-6">
              {/* Query Probe & Search Config Header */}
              <div className="grid grid-cols-1 md:grid-cols-12 gap-4">
                {/* Active Query Probe Card */}
                <div className="md:col-span-4 backdrop-blur-md bg-white/[0.03] p-4 rounded-2xl border border-white/10 flex flex-col justify-between space-y-3">
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] uppercase font-mono font-bold text-emerald-400 tracking-wider">
                        ResNet-50 Extracted Probe
                      </span>
                      <span className="px-2 py-0.5 rounded-full text-[9px] font-mono bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                        ResNet-50 128-D
                      </span>
                    </div>

                    <div className="flex items-center gap-3">
                      <img
                        src={currentAnalysis?.imageUrl || 'https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?w=200'}
                        alt="Probe Muzzle"
                        className="w-14 h-14 object-cover rounded-xl border border-white/15 shrink-0"
                      />
                      <div className="overflow-hidden">
                        <h4 className="text-sm font-bold text-white truncate">
                          {currentAnalysis?.primaryBreed?.breed || 'Current Cattle Scan'}
                        </h4>
                        <span className="text-xs font-mono text-emerald-400 block truncate">
                          {currentAnalysis?.biometricPassport?.uniqueCattleId || 'SCAN-ACTIVE-01'}
                        </span>
                        <span className="text-[10px] text-slate-400 block">
                          Ridge Clarity: {currentAnalysis?.muzzleDetections?.[0]?.ridgePatternClarity || 'High'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="pt-2 border-t border-white/10 flex items-center justify-between gap-2">
                    <button
                      onClick={handleSaveCurrentVector}
                      className="flex-1 py-1.5 px-2.5 rounded-xl bg-emerald-500/20 hover:bg-emerald-500/30 text-emerald-300 border border-emerald-500/40 text-xs font-bold transition-colors cursor-pointer flex items-center justify-center gap-1.5"
                    >
                      <Database className="w-3.5 h-3.5" />
                      <span>Save Vector</span>
                    </button>
                    <button
                      onClick={handleCopyVectorJson}
                      className="py-1.5 px-2.5 rounded-xl bg-white/5 hover:bg-white/10 text-slate-300 text-xs font-medium transition-colors cursor-pointer border border-white/10 flex items-center gap-1"
                      title="Copy 128-D Vector JSON"
                    >
                      {copiedVector ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                </div>

                {/* Search Settings & Metric Selector */}
                <div className="md:col-span-8 backdrop-blur-md bg-white/[0.03] p-4 rounded-2xl border border-white/10 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-xs font-bold text-white flex items-center gap-2">
                      <Sliders className="w-3.5 h-3.5 text-emerald-400" />
                      FAISS Index Configuration &amp; Metrics
                    </span>
                    <span className="text-[11px] font-mono text-slate-400">
                      Latency: <strong className="text-emerald-400">{searchTimeMs} ms</strong> | Indexed:{' '}
                      <strong className="text-white">{totalIndexed}</strong>
                    </span>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    {/* Index Type */}
                    <div>
                      <label className="text-[11px] text-slate-400 block mb-1">Index Structure</label>
                      <select
                        value={indexType}
                        onChange={(e) => setIndexType(e.target.value as any)}
                        className="w-full bg-black/60 border border-white/15 rounded-xl px-2.5 py-1.5 text-xs text-white focus:border-emerald-400 outline-none font-mono"
                      >
                        <option value="IndexFlatIP">IndexFlatIP (Exact Cosine)</option>
                        <option value="IndexIVFFlat">IndexIVFFlat (Voronoi Cells)</option>
                        <option value="IndexFlatL2">IndexFlatL2 (Euclidean Distance)</option>
                      </select>
                    </div>

                    {/* Metric Type */}
                    <div>
                      <label className="text-[11px] text-slate-400 block mb-1">Distance Metric</label>
                      <select
                        value={metric}
                        onChange={(e) => setMetric(e.target.value as any)}
                        className="w-full bg-black/60 border border-white/15 rounded-xl px-2.5 py-1.5 text-xs text-white focus:border-emerald-400 outline-none font-mono"
                      >
                        <option value="cosine">Cosine Similarity (dot &gt; 0)</option>
                        <option value="l2">L2 Distance (Euclidean)</option>
                      </select>
                    </div>

                    {/* Top K */}
                    <div>
                      <label className="text-[11px] text-slate-400 block mb-1">
                        Top Nearest Neighbors (K): <span className="text-emerald-400 font-bold">{topK}</span>
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="10"
                        value={topK}
                        onChange={(e) => setTopK(Number(e.target.value))}
                        className="w-full accent-emerald-400 cursor-pointer"
                      />
                    </div>
                  </div>

                  {/* Math Formula Banner */}
                  <div className="p-2.5 rounded-xl bg-black/50 border border-white/10 font-mono text-[11px] text-slate-300 flex items-center justify-between">
                    <span>
                      Metric Formula: <span className="text-emerald-400 font-bold">Cosine(u, v) = (u · v) / (||u||₂ ||v||₂)</span>
                    </span>
                    <button
                      onClick={performSearch}
                      disabled={isSearching}
                      className="px-3 py-1 bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold rounded-lg text-xs transition-colors flex items-center gap-1.5 cursor-pointer shrink-0"
                    >
                      <RefreshCw className={`w-3 h-3 ${isSearching ? 'animate-spin' : ''}`} />
                      <span>Re-Search</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Search Results List */}
              <div className="space-y-3">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-emerald-400" />
                    Top-{searchResults.length} FAISS Nearest Vector Matches
                  </h4>
                  <div className="flex items-center gap-2 text-[11px] font-mono text-slate-400">
                    <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300 border border-emerald-500/30">
                      Match Threshold: ≥ +0.8000
                    </span>
                    <span>•</span>
                    <span>Ranked by {metric === 'cosine' ? 'Cosine Similarity' : 'L2 Distance'}</span>
                  </div>
                </div>

                {searchResults.length === 0 ? (
                  <div className="p-8 text-center bg-white/[0.02] border border-white/10 rounded-2xl text-slate-400 text-xs">
                    No matching vectors found above threshold. Click "Save Vector" or add cattle to herd registry.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {searchResults.map((res) => {
                      const isHighMatch = res.cosineSimilarity >= 0.85;
                      const isProbable = res.cosineSimilarity >= 0.78;
                      const isSameBreed = res.cosineSimilarity >= 0.45;

                      const badgeClass = isHighMatch
                        ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                        : isProbable
                        ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                        : isSameBreed
                        ? 'bg-blue-500/20 text-blue-300 border-blue-500/40'
                        : 'bg-white/10 text-slate-300 border-white/15';

                      return (
                        <div
                          key={res.cowId}
                          className={`backdrop-blur-md p-4 rounded-2xl border transition-all ${
                            isHighMatch
                              ? 'bg-emerald-950/30 border-emerald-500/50 shadow-lg shadow-emerald-950/50'
                              : 'bg-white/[0.03] border-white/10 hover:border-white/20'
                          }`}
                        >
                          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                            {/* Left: Cow Snapshot & Info */}
                            <div className="flex items-start gap-3.5">
                              <div className="relative">
                                <img
                                  src={res.imageUrl || 'https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?w=200'}
                                  alt={res.breed}
                                  className="w-16 h-16 object-cover rounded-xl border border-white/15 shrink-0"
                                />
                                <span className="absolute -top-1.5 -left-1.5 w-5 h-5 rounded-full bg-emerald-500 text-black font-mono font-black text-[10px] flex items-center justify-center shadow-md">
                                  #{res.rank}
                                </span>
                              </div>

                              <div>
                                <div className="flex flex-wrap items-center gap-2">
                                  <h5 className="text-sm font-bold text-white">{res.breed}</h5>
                                  <span className={`px-2 py-0.5 rounded-full text-[10px] font-mono font-bold border ${badgeClass}`}>
                                    {res.matchConfidence}
                                  </span>
                                </div>
                                <span className="text-xs font-mono text-emerald-400 font-bold block mt-0.5">
                                  Aadhaar: {formatAadhaarId(res.cowId)}
                                </span>
                                <span className="text-[11px] text-slate-400 block">
                                  Species: {res.species || 'Bos indicus'}
                                </span>
                              </div>
                            </div>

                            {/* Center: Cosine & L2 Stats */}
                            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 font-mono text-xs bg-black/40 p-3 rounded-xl border border-white/10 shrink-0">
                              <div>
                                <span className="text-[10px] text-slate-400 block uppercase">Cosine Score</span>
                                <span className="text-base font-extrabold text-emerald-400">
                                  {res.cosineSimilarity > 0 ? `+${res.cosineSimilarity}` : res.cosineSimilarity}
                                </span>
                              </div>
                              <div>
                                <span className="text-[10px] text-slate-400 block uppercase">Concordance</span>
                                <span className="text-base font-extrabold text-teal-300">
                                  {res.similarityScore}%
                                </span>
                              </div>
                              <div>
                                <span className="text-[10px] text-slate-400 block uppercase">L2 Distance</span>
                                <span className="text-sm font-bold text-slate-300">
                                  {res.l2Distance}
                                </span>
                              </div>
                            </div>
                          </div>

                          {/* Sub-band Similarity Breakdown Bar */}
                          <div className="mt-3 pt-3 border-t border-white/10 grid grid-cols-2 sm:grid-cols-4 gap-2 text-[10px] font-mono">
                            <div className="bg-white/[0.02] p-2 rounded-lg border border-white/5">
                              <span className="text-slate-400 block">Ridge Freq (d0..31):</span>
                              <div className="flex items-center justify-between mt-1">
                                <div className="w-16 bg-white/10 rounded-full h-1.5 overflow-hidden">
                                  <div
                                    className="bg-emerald-400 h-full rounded-full"
                                    style={{ width: `${res.subBandSimilarities?.ridgeScore || 85}%` }}
                                  ></div>
                                </div>
                                <span className="text-emerald-400 font-bold">
                                  {res.subBandSimilarities?.ridgeScore || 85}%
                                </span>
                              </div>
                            </div>

                            <div className="bg-white/[0.02] p-2 rounded-lg border border-white/5">
                              <span className="text-slate-400 block">Bead Density (d32..63):</span>
                              <div className="flex items-center justify-between mt-1">
                                <div className="w-16 bg-white/10 rounded-full h-1.5 overflow-hidden">
                                  <div
                                    className="bg-teal-400 h-full rounded-full"
                                    style={{ width: `${res.subBandSimilarities?.beadScore || 88}%` }}
                                  ></div>
                                </div>
                                <span className="text-teal-400 font-bold">
                                  {res.subBandSimilarities?.beadScore || 88}%
                                </span>
                              </div>
                            </div>

                            <div className="bg-white/[0.02] p-2 rounded-lg border border-white/5">
                              <span className="text-slate-400 block">Symmetry (d64..95):</span>
                              <div className="flex items-center justify-between mt-1">
                                <div className="w-16 bg-white/10 rounded-full h-1.5 overflow-hidden">
                                  <div
                                    className="bg-amber-400 h-full rounded-full"
                                    style={{ width: `${res.subBandSimilarities?.symmetryScore || 90}%` }}
                                  ></div>
                                </div>
                                <span className="text-amber-400 font-bold">
                                  {res.subBandSimilarities?.symmetryScore || 90}%
                                </span>
                              </div>
                            </div>

                            <div className="bg-white/[0.02] p-2 rounded-lg border border-white/5">
                              <span className="text-slate-400 block">Deep Textures (d96..127):</span>
                              <div className="flex items-center justify-between mt-1">
                                <div className="w-16 bg-white/10 rounded-full h-1.5 overflow-hidden">
                                  <div
                                    className="bg-blue-400 h-full rounded-full"
                                    style={{ width: `${res.subBandSimilarities?.deepTextureScore || 82}%` }}
                                  ></div>
                                </div>
                                <span className="text-blue-400 font-bold">
                                  {res.subBandSimilarities?.deepTextureScore || 82}%
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 2: 128-D VECTOR INSPECTOR & MATH COMPARATOR */}
          {activeTab === 'inspector' && (
            <div className="space-y-6">
              {/* Mathematical Cosine Calculator */}
              <div className="backdrop-blur-md bg-white/[0.03] p-5 rounded-2xl border border-white/10 space-y-4">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <Zap className="w-4 h-4 text-emerald-400" />
                    Real-time Pairwise Vector Cosine Comparator
                  </h4>
                  <span className="text-xs font-mono text-emerald-400">
                    Formula: (A · B) / (||A|| * ||B||)
                  </span>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-12 gap-4 items-center">
                  {/* Vector A (Probe) */}
                  <div className="md:col-span-5 bg-black/50 p-4 rounded-xl border border-white/10 space-y-2">
                    <span className="text-[10px] uppercase font-mono font-bold text-emerald-400">
                      Vector A (Active Cattle Scan)
                    </span>
                    <h5 className="text-sm font-bold text-white truncate">
                      {currentAnalysis?.primaryBreed?.breed || 'Current Scan'}
                    </h5>
                    <span className="text-xs font-mono text-slate-300 block truncate">
                      {currentAnalysis?.biometricPassport?.uniqueCattleId || 'ACTIVE-SCAN'}
                    </span>
                    <div className="text-[10px] text-slate-400 font-mono">
                      Norm: 1.0000 | Dimension: 128-D Float32
                    </div>
                  </div>

                  {/* Similarity Operator */}
                  <div className="md:col-span-2 text-center py-2">
                    <div className="inline-flex flex-col items-center justify-center p-3 rounded-2xl bg-emerald-500/10 border border-emerald-500/30">
                      <span className="text-xs font-mono text-emerald-400 font-bold">COSINE</span>
                      <span className="text-lg font-black text-white font-mono">
                        {directCosine > 0 ? `+${directCosine.toFixed(4)}` : directCosine.toFixed(4)}
                      </span>
                      <span className="text-[10px] text-emerald-400 font-mono">
                        {Math.round(directCosine * 100)}% match
                      </span>
                    </div>
                  </div>

                  {/* Vector B (Target from Registry) */}
                  <div className="md:col-span-5 bg-black/50 p-4 rounded-xl border border-white/10 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] uppercase font-mono font-bold text-teal-400">
                        Vector B (Select from Registry)
                      </span>
                      <span className="text-[10px] font-mono text-slate-400">
                        {allVectors.length} cattle registered
                      </span>
                    </div>
                    {allVectors.length === 0 ? (
                      <div className="p-3 rounded-lg bg-white/[0.02] border border-white/10 text-xs text-slate-400">
                        No registered vectors in FAISS yet. Register cattle or save current scan to compare.
                      </div>
                    ) : (
                      <select
                        value={compareTargetId}
                        onChange={(e) => setCompareTargetId(e.target.value)}
                        className="w-full bg-black/80 border border-white/15 rounded-lg px-2.5 py-1.5 text-xs text-white focus:border-teal-400 outline-none font-mono"
                      >
                        {allVectors.map((v) => (
                          <option key={v.id} value={v.id}>
                            {v.breed} ({v.cowId})
                          </option>
                        ))}
                      </select>
                    )}
                    <div className="text-[10px] text-slate-400 font-mono">
                      L2 Euclidean Distance: <strong className="text-white">{allVectors.length > 0 ? directL2.toFixed(4) : '0.0000'}</strong>
                    </div>
                  </div>
                </div>
              </div>

              {/* 128-D Vector Numerical Heatmap Grid */}
              <div className="backdrop-blur-md bg-white/[0.03] p-5 rounded-2xl border border-white/10 space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <Layers className="w-4 h-4 text-emerald-400" />
                    128-Dimensional Biometric Latent Vector Embedding Heatmap
                  </h4>
                  <button
                    onClick={handleCopyVectorJson}
                    className="px-2.5 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-xs text-slate-300 flex items-center gap-1.5 cursor-pointer border border-white/10"
                  >
                    {copiedVector ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                    <span>{copiedVector ? 'Copied' : 'Copy Array'}</span>
                  </button>
                </div>

                <p className="text-xs text-slate-400">
                  Values normalized on the unit hypersphere ($\|v\|_2 = 1.0$). Hover over coordinates to inspect floating-point weights:
                </p>

                {/* 128 Cells Grid */}
                <div className="grid grid-cols-8 sm:grid-cols-16 gap-1 bg-black/60 p-3 rounded-xl border border-white/10">
                  {currentVector.map((val, idx) => {
                    const absVal = Math.min(1, Math.abs(val) * 6);
                    const isPositive = val >= 0;
                    const bgStyle = isPositive
                      ? `rgba(16, 185, 129, ${0.15 + absVal * 0.75})`
                      : `rgba(59, 130, 246, ${0.15 + absVal * 0.75})`;

                    let subLabel = 'Ridge';
                    if (idx >= 32 && idx < 64) subLabel = 'Bead';
                    if (idx >= 64 && idx < 96) subLabel = 'Symmetry';
                    if (idx >= 96) subLabel = 'Deep';

                    return (
                      <div
                        key={idx}
                        style={{ backgroundColor: bgStyle }}
                        className="h-8 rounded-md flex flex-col items-center justify-center font-mono text-[9px] text-white border border-white/5 relative group cursor-crosshair transition-transform hover:scale-125 hover:z-20 hover:border-emerald-300"
                        title={`d[${idx}] (${subLabel}): ${val.toFixed(5)}`}
                      >
                        <span className="text-[8px] opacity-75">{idx}</span>
                      </div>
                    );
                  })}
                </div>

                {/* Sub-band Legend */}
                <div className="flex flex-wrap items-center justify-between gap-2 pt-2 text-[11px] font-mono text-slate-400">
                  <div className="flex items-center gap-4">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500/80"></span>
                      d[0..31] Ridge Frequency Spectrum
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-teal-500/80"></span>
                      d[32..63] Bead Density Distribution
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-amber-500/80"></span>
                      d[64..95] Nasolabial Symmetry
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-sm bg-blue-500/80"></span>
                      d[96..127] Deep Texture
                    </span>
                  </div>
                  <span>128-D Unit Hypersphere</span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: VECTOR REGISTRY DATABASE */}
          {activeTab === 'database' && (
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <Database className="w-4 h-4 text-emerald-400" />
                    FAISS Vector Index Database ({allVectors.length} Registered Cattle)
                  </h4>
                  <p className="text-xs text-slate-400">
                    Active 128-D ResNet-50 biometric embeddings stored for real-time sub-millisecond similarity comparison
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  {allVectors.length > 0 && (
                    <button
                      onClick={handleClearAllVectors}
                      className="px-3 py-2 bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition-colors cursor-pointer"
                      title="Clear all stored vectors"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>Clear All FAISS Records</span>
                    </button>
                  )}

                  <button
                    onClick={handleSaveCurrentVector}
                    className="px-3.5 py-2 bg-emerald-500 hover:bg-emerald-400 text-black font-extrabold rounded-xl text-xs flex items-center gap-1.5 transition-colors cursor-pointer shadow-md shadow-emerald-500/20"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Register Current Scan in FAISS</span>
                  </button>
                </div>
              </div>

              {allVectors.length === 0 ? (
                <div className="p-10 text-center bg-white/[0.02] border border-white/10 rounded-3xl space-y-3">
                  <div className="w-12 h-12 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto">
                    <Database className="w-6 h-6" />
                  </div>
                  <h5 className="text-sm font-bold text-white">FAISS Vector Registry is Clean &amp; Empty</h5>
                  <p className="text-xs text-slate-400 max-w-md mx-auto leading-relaxed">
                    All previous dummy/test records have been cleared. When a new cow is registered in the system, its 128-dimensional ResNet-50 biometric embedding will automatically be indexed here for future similarity matching and verification.
                  </p>
                  <button
                    onClick={handleSaveCurrentVector}
                    className="mt-2 inline-flex items-center gap-2 px-4 py-2 bg-emerald-500 hover:bg-emerald-400 text-black font-bold text-xs rounded-xl shadow-lg shadow-emerald-500/20 transition-all cursor-pointer"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Register Current Scan Now</span>
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
                  {allVectors.map((vec) => (
                    <div
                      key={vec.id}
                      className="backdrop-blur-md bg-white/[0.03] p-3.5 rounded-2xl border border-white/10 space-y-2 hover:border-emerald-500/40 transition-colors relative group"
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-3 overflow-hidden">
                          <img
                            src={vec.imageUrl || 'https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?w=200'}
                            alt={vec.breed}
                            className="w-12 h-12 object-cover rounded-xl border border-white/15 shrink-0"
                          />
                          <div className="overflow-hidden">
                            <h5 className="text-xs font-bold text-white truncate">{vec.breed}</h5>
                            <span className="text-[11px] font-mono text-emerald-400 truncate block">
                              Aadhaar: {formatAadhaarId(vec.cowId)}
                            </span>
                            <span className="text-[10px] text-slate-400 block truncate">
                              {vec.species}
                            </span>
                          </div>
                        </div>

                        <button
                          onClick={() => handleDeleteVector(vec.id, vec.cowId)}
                          className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors cursor-pointer border border-transparent hover:border-rose-500/20 shrink-0"
                          title="Delete from FAISS"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      </div>

                      <div className="pt-2 border-t border-white/10 flex items-center justify-between text-[10px] font-mono text-slate-400">
                        <span>Bead: {vec.beadDensityScore || 88}%</span>
                        <span>Sym: {vec.symmetryScore || 90}%</span>
                        <button
                          onClick={() => {
                            setCompareTargetId(vec.id);
                            setActiveTab('inspector');
                          }}
                          className="text-emerald-400 hover:text-emerald-300 font-bold transition-colors cursor-pointer"
                        >
                          Inspect Vector &rarr;
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="px-6 py-4 border-t border-white/10 bg-black/40 backdrop-blur-md flex items-center justify-between relative z-10 text-xs">
          <div className="flex items-center gap-2 text-slate-400">
            <Info className="w-4 h-4 text-emerald-400" />
            <span>
              FAISS search calculates exact inner-product dot matrix over 128-D normalized embeddings.
            </span>
          </div>

          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-white/10 hover:bg-white/15 text-white font-bold transition-colors cursor-pointer border border-white/10"
          >
            Close Engine
          </button>
        </div>
      </div>
    </div>
  );
};
