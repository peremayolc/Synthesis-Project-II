import { ReactNode } from "react";
import { Link, useLocation } from "wouter";
import { FilePlus, FolderOpen, Users, ShieldCheck, Sun, Moon } from "lucide-react";
import { useTheme } from "next-themes";
import { Button } from "@/components/ui/button";

const NAV_ITEMS = [
  { href: "/", label: "New Translation", icon: FilePlus },
  { href: "/projects", label: "Projects", icon: FolderOpen },
  { href: "/clients", label: "Clients", icon: Users },
  { href: "/qa", label: "QA", icon: ShieldCheck },
];

export function Layout({ children }: { children: ReactNode }) {
  const [location] = useLocation();
  const { theme, setTheme } = useTheme();

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      {/* Sidebar */}
      <div className="w-64 border-r border-border bg-sidebar flex flex-col justify-between shrink-0">
        <div>
          <div className="p-6">
            <div className="text-2xl font-bold tracking-tight">
              <span className="text-primary">i</span>
              <span className="text-foreground">DISC</span>
            </div>
            <div className="mt-1">
              <h1 className="text-xl font-semibold text-foreground">Lexor</h1>
              <p className="text-xs text-muted-foreground mt-0.5">Domain Translation Assistant</p>
            </div>
          </div>

          <nav className="px-3 py-2 space-y-1">
            {NAV_ITEMS.map((item) => {
              const isActive = location === item.href;
              return (
                <Link key={item.href} href={item.href} className={`flex items-center gap-3 px-3 py-2.5 rounded-md transition-colors ${isActive ? 'bg-accent text-accent-foreground font-medium' : 'text-sidebar-foreground hover:bg-muted'}`}>
                  <item.icon className={`h-5 w-5 ${isActive ? 'text-primary' : 'text-muted-foreground'}`} />
                  {item.label}
                </Link>
              );
            })}
          </nav>
        </div>

        <div className="p-6">
          <div className="inline-flex items-center justify-center px-2.5 py-1 text-xs font-medium rounded-full bg-muted text-muted-foreground">
            Lexor v2.4
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Topbar */}
        <header className="h-16 border-b border-border bg-background flex items-center justify-between px-6 shrink-0">
          <div className="font-semibold text-lg">
            {NAV_ITEMS.find(i => i.href === location)?.label || "Dashboard"}
          </div>
          
          <div className="flex items-center gap-4">
            <Button variant="ghost" size="icon" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} className="text-muted-foreground hover:text-foreground">
              <Sun className="h-5 w-5 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <Moon className="absolute h-5 w-5 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
            
            <div className="h-8 w-px bg-border mx-1"></div>

            <div className="flex items-center gap-3">
              <div className="text-right hidden sm:block">
                <div className="text-sm font-medium text-foreground">Translator</div>
                <div className="text-xs text-primary font-medium">Pro User</div>
              </div>
              <div className="h-9 w-9 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center text-primary font-bold text-sm">
                TR
              </div>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-auto bg-card p-6 md:p-8">
          <div className="max-w-6xl mx-auto h-full">
            {children}
          </div>
        </main>
      </div>
    </div>
  );
}