import { motion } from "framer-motion";
import { BookA, MessageSquareQuote, AlignLeft, UserCheck } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const features = [
  {
    title: "Terminology Check",
    description: "Ensures strict compliance with client glossaries and industry-standard domain terms. Flags any deviations before human review.",
    icon: BookA,
    status: "Active",
    statusColor: "green"
  },
  {
    title: "Meaning Preservation",
    description: "Deep semantic analysis to ensure the target text carries the exact nuance and legal/technical weight of the source.",
    icon: MessageSquareQuote,
    status: "Active",
    statusColor: "green"
  },
  {
    title: "Formatting Consistency",
    description: "Verifies that all tags, numbered lists, bullet points, and bold/italic styling perfectly matches the source document.",
    icon: AlignLeft,
    status: "Active",
    statusColor: "green"
  },
  {
    title: "Human Escalation",
    description: "Smart routing system that identifies highly ambiguous or legally critical segments and automatically assigns them to senior translators.",
    icon: UserCheck,
    status: "Demo Mode",
    statusColor: "amber"
  }
];

export default function QA() {
  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-8"
    >
      <div>
        <h1 className="text-2xl font-bold tracking-tight">QA Review</h1>
        <p className="text-muted-foreground mt-1">Second AI layer for error detection and correction</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {features.map((feature, i) => (
          <motion.div
            key={feature.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
          >
            <Card className="p-6 h-full flex flex-col relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-6">
                <Badge 
                  className={
                    feature.statusColor === 'green' 
                      ? 'bg-green-100 text-green-800 border-green-200 dark:bg-green-900/30 dark:text-green-400 dark:border-green-800' 
                      : 'bg-amber-100 text-amber-800 border-amber-200 dark:bg-amber-900/30 dark:text-amber-400 dark:border-amber-800'
                  }
                >
                  {feature.status}
                </Badge>
              </div>
              
              <div className="p-3 bg-primary/10 w-fit rounded-lg text-primary mb-4 group-hover:scale-110 transition-transform">
                <feature.icon className="w-6 h-6" />
              </div>
              
              <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
              <p className="text-muted-foreground leading-relaxed">
                {feature.description}
              </p>
            </Card>
          </motion.div>
        ))}
      </div>

      <div className="bg-primary/5 border border-primary/20 rounded-lg p-4 text-center">
        <p className="text-sm font-medium text-primary">
          QA module demo handled separately. This view shows active protection layers.
        </p>
      </div>
    </motion.div>
  );
}