"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

const LINKS = [
  { href: "/", label: "Player lookup" },
  { href: "/rankings", label: "Rankings" },
];

export default function Sidebar() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  // Close the mobile menu after navigating.
  useEffect(() => setOpen(false), [pathname]);

  return (
    <header className="sidebar">
      <Link href="/" className="sidebar-brand">
        Projections
      </Link>
      <button
        type="button"
        className="sidebar-toggle"
        aria-expanded={open}
        aria-controls="site-nav"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "Close" : "Menu"}
      </button>
      <nav id="site-nav" className="sidebar-nav" data-open={open} aria-label="Main">
        <ul className="nav-list">
          {LINKS.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                className="nav-link"
                aria-current={pathname === link.href ? "page" : undefined}
              >
                {link.label}
              </Link>
            </li>
          ))}
        </ul>
      </nav>
    </header>
  );
}
