"use client";
import * as DialogPrimitive from "@radix-ui/react-dialog";
import { X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { usePathname } from "next/navigation";

// Right-side sheet built on Radix Dialog. Used for the mobile nav panel,
// workspace settings, and run-details/metadata panels. Closes itself on
// route change so navigating away doesn't leave it open over the new page.
export function Drawer({
  title,
  description,
  trigger,
  children,
}: {
  title: string;
  description?: string;
  trigger: ReactNode;
  children: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  useEffect(() => setOpen(false), [pathname]);
  return (
    <DialogPrimitive.Root open={open} onOpenChange={setOpen}>
      <DialogPrimitive.Trigger asChild>{trigger}</DialogPrimitive.Trigger>
      <DialogPrimitive.Portal>
        <DialogPrimitive.Overlay className="fixed inset-0 z-40 bg-black/25 backdrop-blur-[2px] data-[state=closed]:opacity-0 data-[state=open]:opacity-100 transition-opacity duration-200 ease-out" />
        <DialogPrimitive.Content className="fixed inset-y-0 end-0 z-50 w-full max-w-md overflow-y-auto border-s bg-card p-6 shadow-xl data-[state=closed]:translate-x-full data-[state=open]:translate-x-0 transition-transform duration-200 ease-out rtl:data-[state=closed]:-translate-x-full">
          <DialogPrimitive.Title className="font-semibold text-lg pe-8">
            {title}
          </DialogPrimitive.Title>
          <DialogPrimitive.Description className="mt-1 text-sm text-muted-foreground">
            {description ?? "Details for the selected analytical context."}
          </DialogPrimitive.Description>
          <DialogPrimitive.Close
            className="absolute end-4 top-4 rounded-md p-2 hover:bg-muted focus-visible:ring-2"
            aria-label="Close details"
          >
            <X size={18} />
          </DialogPrimitive.Close>
          <div className="mt-6">{children}</div>
        </DialogPrimitive.Content>
      </DialogPrimitive.Portal>
    </DialogPrimitive.Root>
  );
}
