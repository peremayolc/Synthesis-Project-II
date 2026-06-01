import { useState, useEffect } from "react";
import { useLocation } from "wouter";
import { motion, AnimatePresence } from "framer-motion";
import { initialSegments } from "@/data/fakeData";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { CheckCircle2, AlertTriangle, AlertCircle, ArrowRight, User } from "lucide-react";

export default function TranslationResults() {
  const [, setLocation] = useLocation();
  const [segments, setSegments] = useState(initialSegments);
  const [selectedSegmentId, setSelectedSegmentId] = useState(segments[0]?.id);
  const [filter, setFilter] = useState("All");

  const [barsAnimated, setBarsAnimated] = useState(false);
  useEffect(() => {
    setBarsAnimated(true);
  }, []);

  const selectedSegment = segments.find(s => s.id === selectedSegmentId);

  const filteredSegments = segments.filter(s => {
    if (filter === "All") return true;
    if (filter === "Green") return s.status === "green";
    if (filter === "Yellow") return s.status === "yellow";
    if (filter === "Red") return s.status === "red";
    if (filter === "Flagged") return s.status === "human";
    return true;
  });

  const getStatusColor = (status: string) => {
    switch (status) {
      case "green": return "bg-green-500";
      case "yellow": return "bg-yellow-500";
      case "red": return "bg-red-500";
      case "human": return "bg-blue-500";
      default: return "bg-gray-500";
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "green": return <Badge className="bg-green-100 text-green-800 hover:bg-green-100 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800"><CheckCircle2 className="w-3 h-3 mr-1" /> Verified</Badge>;
      case "yellow": return <Badge className="bg-yellow-100 text-yellow-800 hover:bg-yellow-100 border-yellow-200 dark:bg-yellow-900/30 dark:text-yellow-400 dark:border-yellow-800"><AlertTriangle className="w-3 h-3 mr-1" /> Needs Review</Badge>;
      case "red": return <Badge className="bg-red-100 text-red-800 hover:bg-red-100 border-red-200 dark:bg-red-900/30 dark:text-red-400 dark:border-red-800"><AlertCircle className="w-3 h-3 mr-1" /> Critical Issue</Badge>;
      case "human": return <Badge className="bg-blue-100 text-blue-800 hover:bg-blue-100 border-blue-200 dark:bg-blue-900/30 dark:text-blue-400 dark:border-blue-800"><User className="w-3 h-3 mr-1" /> Sent to Human</Badge>;
      default: return null;
    }
  };

  const handleApplyFix = () => {
    if (!selectedSegment || !selectedSegment.aifix) return;
    setSegments(prev => prev.map(s => 
      s.id === selectedSegment.id 
        ? { ...s, target: s.aifix!, status: "green", issues: undefined, aifix: undefined } 
        : s
    ));
  };

  const handleMarkVerified = () => {
    if (!selectedSegment) return;
    setSegments(prev => prev.map(s => 
      s.id === selectedSegment.id ? { ...s, status: "green" } : s
    ));
  };

  const handleSendToHuman = () => {
    if (!selectedSegment) return;
    setSegments(prev => prev.map(s => 
      s.id === selectedSegment.id ? { ...s, status: "human" } : s
    ));
  };

  const handleTargetChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    if (!selectedSegment) return;
    setSegments(prev => prev.map(s => 
      s.id === selectedSegment.id ? { ...s, target: e.target.value } : s
    ));
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6 h-full flex flex-col"
    >
      <div className="flex items-start justify-between shrink-0">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Translation Results</h1>
          <p className="text-muted-foreground mt-1 font-mono text-sm">Legal_Contract_Mar2026 • ACME Legal</p>
        </div>
        <Button onClick={() => setLocation("/")}>New Project</Button>
      </div>

      <Card className="p-5 shrink-0 flex flex-col md:flex-row gap-6">
        <div className="w-full md:w-1/3 space-y-4">
          <div className="flex items-center gap-2">
            <Badge className="bg-amber-100 text-amber-800 border-amber-200 hover:bg-amber-100 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-800">Needs Review</Badge>
            <span className="text-sm font-medium">Domain: Legal</span>
          </div>
          
          <div className="space-y-1.5">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Confidence Score</span>
              <span className="font-semibold">High</span>
            </div>
            <div className="h-3 w-full bg-muted rounded-full overflow-hidden flex">
              <motion.div 
                className="h-full bg-green-500"
                initial={{ width: "0%" }}
                animate={{ width: barsAnimated ? "82%" : "0%" }}
                transition={{ duration: 1, ease: "easeOut" }}
              />
              <motion.div 
                className="h-full bg-yellow-500"
                initial={{ width: "0%" }}
                animate={{ width: barsAnimated ? "13%" : "0%" }}
                transition={{ duration: 1, ease: "easeOut", delay: 0.2 }}
              />
              <motion.div 
                className="h-full bg-red-500"
                initial={{ width: "0%" }}
                animate={{ width: barsAnimated ? "5%" : "0%" }}
                transition={{ duration: 1, ease: "easeOut", delay: 0.4 }}
              />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground pt-1">
              <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-green-500 mr-1.5"></span>82% Valid</span>
              <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-yellow-500 mr-1.5"></span>13% Review</span>
              <span className="flex items-center"><span className="w-2 h-2 rounded-full bg-red-500 mr-1.5"></span>5% Critical</span>
            </div>
          </div>
        </div>

        <div className="hidden md:block w-px bg-border"></div>

        <div className="w-full md:w-2/3">
          <h4 className="text-sm font-semibold mb-3">Detected Issues</h4>
          <ul className="space-y-2">
            <li className="flex items-start gap-2 text-sm">
              <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5 shrink-0" />
              <span>Terminology inconsistency (Legal glossary constraint)</span>
            </li>
            <li className="flex items-start gap-2 text-sm">
              <AlertCircle className="w-4 h-4 text-red-500 mt-0.5 shrink-0" />
              <span>Meaning deviation in complex clauses</span>
            </li>
            <li className="flex items-start gap-2 text-sm">
              <AlertTriangle className="w-4 h-4 text-yellow-500 mt-0.5 shrink-0" />
              <span>Formatting risk in numbered lists</span>
            </li>
          </ul>
        </div>
      </Card>

      <div className="flex-1 flex gap-6 min-h-0">
        {/* Left Column: Segments List */}
        <div className="w-2/5 flex flex-col bg-background border rounded-lg overflow-hidden">
          <div className="p-3 border-b flex gap-2 overflow-x-auto shrink-0">
            {["All", "Green", "Yellow", "Red", "Flagged"].map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1.5 text-xs font-medium rounded-full whitespace-nowrap transition-colors ${filter === f ? 'bg-primary text-primary-foreground' : 'bg-muted text-muted-foreground hover:bg-muted/80'}`}
              >
                {f}
              </button>
            ))}
          </div>
          
          <div className="flex-1 overflow-y-auto">
            <AnimatePresence>
              {filteredSegments.map((segment, idx) => (
                <motion.div
                  key={segment.id}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: idx * 0.05 }}
                  onClick={() => setSelectedSegmentId(segment.id)}
                  className={`p-4 border-b cursor-pointer transition-colors relative ${selectedSegmentId === segment.id ? 'bg-muted/50' : 'hover:bg-muted/30'}`}
                >
                  {selectedSegmentId === segment.id && (
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary"></div>
                  )}
                  <div className="flex items-start gap-3">
                    <span className="text-xs text-muted-foreground font-mono mt-0.5 w-4">{segment.id}</span>
                    <div className={`w-2 h-2 rounded-full mt-1.5 shrink-0 ${getStatusColor(segment.status)}`}></div>
                    <p className="text-sm font-medium line-clamp-2 leading-relaxed">{segment.source}</p>
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        </div>

        {/* Right Column: Segment Detail */}
        <div className="w-3/5 bg-background border rounded-lg flex flex-col">
          {selectedSegment ? (
            <div className="flex-1 flex flex-col overflow-y-auto p-6 space-y-6">
              <div className="flex items-center justify-between pb-4 border-b">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-sm text-muted-foreground">Segment {selectedSegment.id}</span>
                  {getStatusBadge(selectedSegment.status)}
                </div>
              </div>

              <div className="space-y-3">
                <label className="text-xs font-bold tracking-wider text-muted-foreground">SOURCE</label>
                <div className="p-4 bg-muted/30 rounded-md text-base leading-relaxed border border-transparent">
                  {selectedSegment.source}
                </div>
              </div>

              <div className="space-y-3">
                <label className="text-xs font-bold tracking-wider text-muted-foreground">TARGET</label>
                <Textarea 
                  value={selectedSegment.target}
                  onChange={handleTargetChange}
                  className="min-h-[120px] text-base leading-relaxed resize-none focus-visible:ring-primary/50"
                />
              </div>

              {selectedSegment.issues && selectedSegment.issues.length > 0 && (
                <div className="space-y-3 p-4 bg-red-50/50 dark:bg-red-950/20 border border-red-100 dark:border-red-900/50 rounded-md">
                  <label className="text-xs font-bold tracking-wider text-red-800 dark:text-red-400 flex items-center gap-1">
                    <AlertCircle className="w-3.5 h-3.5" /> ISSUES DETECTED
                  </label>
                  <ul className="space-y-2">
                    {selectedSegment.issues.map((issue, i) => (
                      <li key={i} className="text-sm text-red-900 dark:text-red-300 ml-4 list-disc">{issue}</li>
                    ))}
                  </ul>
                </div>
              )}

              {selectedSegment.aifix && (
                <div className="space-y-3 p-4 bg-primary/5 border border-primary/20 rounded-md">
                  <label className="text-xs font-bold tracking-wider text-primary flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> AI SUGGESTED FIX
                  </label>
                  <p className="text-sm font-medium">{selectedSegment.aifix}</p>
                </div>
              )}

              <div className="mt-auto pt-6 border-t flex flex-wrap gap-3">
                {selectedSegment.aifix && (
                  <Button onClick={handleApplyFix} className="gap-2">
                    Apply AI Fix
                  </Button>
                )}
                <Button variant="outline" onClick={handleMarkVerified} className="gap-2 border-green-200 hover:bg-green-50 hover:text-green-700 dark:border-green-900/50 dark:hover:bg-green-900/20 dark:hover:text-green-400">
                  <CheckCircle2 className="w-4 h-4" /> Mark Verified
                </Button>
                <Button variant="outline" onClick={handleSendToHuman} className="gap-2 ml-auto">
                  <User className="w-4 h-4" /> Send to Human
                </Button>
              </div>
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center text-muted-foreground">
              Select a segment to view details
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}